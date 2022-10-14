from rlpyt.replays.non_sequence.frame import (
    AsyncPrioritizedReplayFrameBuffer,
    AsyncUniformReplayFrameBuffer,
    PrioritizedReplayFrameBuffer,
    UniformReplayFrameBuffer,
)
from rlpyt.replays.non_sequence.prioritized import (
    AsyncPrioritizedReplayBuffer,
    PrioritizedReplayBuffer,
)
from rlpyt.replays.non_sequence.time_limit import (
    AsyncTlPrioritizedReplayBuffer,
    AsyncTlUniformReplayBuffer,
    TlPrioritizedReplayBuffer,
    TlUniformReplayBuffer,
)
from rlpyt.replays.non_sequence.uniform import (
    AsyncUniformReplayBuffer,
    UniformReplayBuffer,
)
from rlpyt.replays.sequence.frame import (
    AsyncPrioritizedSequenceReplayFrameBuffer,
    AsyncUniformSequenceReplayFrameBuffer,
    PrioritizedSequenceReplayFrameBuffer,
    UniformSequenceReplayFrameBuffer,
)
from rlpyt.replays.sequence.prioritized import (
    AsyncPrioritizedSequenceReplayBuffer,
    PrioritizedSequenceReplayBuffer,
)
from rlpyt.replays.sequence.uniform import (
    AsyncUniformSequenceReplayBuffer,
    UniformSequenceReplayBuffer,
)
from rlpyt.utils.logging import logger

from chloe.utils.replay_components import (
    AugReplayBufferExtractMixin,
    AugSeqReplayBufferExtractMixin,
)


class AugTlUniformReplayBuffer(AugReplayBufferExtractMixin, TlUniformReplayBuffer):
    """Overloads the `TlUniformReplayBuffer` class with custom functions.

    TlUniformReplayBuffer with custom functions dedicated to fetch data from
    the replay buffer.

    The `extract_batch` method is overloaded to retrieve additional information
    such as the `simulated_pathology`, `simulated_patients`, `simulated_severity`
    and eventually the `intermediate_states` in cases `n_step_return` > 1.
    """

    def __init__(
        self, *args, intermediate_data_flag=False, **kwargs,
    ):
        """Instantiates a class object.

        Parameters
        ----------
        intermediate_data_flag: bool
            Flag for extracting intermediate state data from replay buffer.
            default: False

        """
        super().__init__(*args, **kwargs)
        self._set_intermediate_data_flag(intermediate_data_flag)


class AugTlPrioritizedReplayBuffer(
    AugReplayBufferExtractMixin, TlPrioritizedReplayBuffer
):
    """Overloads the `TlPrioritizedReplayBuffer` class with custom functions.

    TlPrioritizedReplayBuffer with custom functions dedicated to fetch data from
    the replay buffer.

    The `extract_batch` method is overloaded to retrieve additional information
    such as the `simulated_pathology`, `simulated_patients`, `simulated_severity`
    and eventually the `intermediate_states` in cases `n_step_return` > 1.
    """

    def __init__(
        self, *args, intermediate_data_flag=False, **kwargs,
    ):
        """Instantiates a class object.

        Parameters
        ----------
        intermediate_data_flag: bool
            Flag for extracting intermediate state data from replay buffer.
            default: False

        """
        super().__init__(*args, **kwargs)
        self._set_intermediate_data_flag(intermediate_data_flag)


class AugAsyncTlUniformReplayBuffer(
    AugReplayBufferExtractMixin, AsyncTlUniformReplayBuffer
):
    """Overloads the `AsyncTlUniformReplayBuffer` class with custom functions.

    AsyncTlUniformReplayBuffer with custom functions dedicated to fetch data from
    the replay buffer.

    The `extract_batch` method is overloaded to retrieve additional information
    such as the `simulated_pathology`, `simulated_patients`, `simulated_severity`
    and eventually the `intermediate_states` in cases `n_step_return` > 1.
    """

    def __init__(
        self, *args, intermediate_data_flag=False, **kwargs,
    ):
        """Instantiates a class object.

        Parameters
        ----------
        intermediate_data_flag: bool
            Flag for extracting intermediate state data from replay buffer.
            default: False

        """
        super().__init__(*args, **kwargs)
        self._set_intermediate_data_flag(intermediate_data_flag)


class AugAsyncTlPrioritizedReplayBuffer(
    AugReplayBufferExtractMixin, AsyncTlPrioritizedReplayBuffer
):
    """Overloads the `AsyncTlPrioritizedReplayBuffer` class with custom functions.

    AsyncTlPrioritizedReplayBuffer with custom functions dedicated
    to fetch data from the replay buffer.

    The `extract_batch` method is overloaded to retrieve additional information
    such as the `simulated_pathology`, `simulated_patients`, `simulated_severity`
    and eventually the `intermediate_states` in cases `n_step_return` > 1.
    """

    def __init__(
        self, *args, intermediate_data_flag=False, **kwargs,
    ):
        """Instantiates a class object.

        Parameters
        ----------
        intermediate_data_flag: bool
            Flag for extracting intermediate state data from replay buffer.
            default: False

        """
        super().__init__(*args, **kwargs)
        self._set_intermediate_data_flag(intermediate_data_flag)


class AugPrioritizedReplayBuffer(AugReplayBufferExtractMixin, PrioritizedReplayBuffer):
    """Overloads the `PrioritizedReplayBuffer` class with custom functions.

    PrioritizedReplayBuffer with custom functions dedicated to fetch data from
    the replay buffer.

    The `extract_batch` method is overloaded to retrieve additional information
    such as the `simulated_pathology`, `simulated_patients`, `simulated_severity`
    and eventually the `intermediate_states` in cases `n_step_return` > 1.
    """

    def __init__(
        self, *args, intermediate_data_flag=False, **kwargs,
    ):
        """Instantiates a class object.

        Parameters
        ----------
        intermediate_data_flag: bool
            Flag for extracting intermediate state data from replay buffer.
            default: False

        """
        super().__init__(*args, **kwargs)
        self._set_intermediate_data_flag(intermediate_data_flag)


class AugAsyncPrioritizedReplayBuffer(
    AugReplayBufferExtractMixin, AsyncPrioritizedReplayBuffer
):
    """Overloads the `AsyncPrioritizedReplayBuffer` class with custom functions.

    AsyncPrioritizedReplayBuffer with custom functions dedicated to fetch data from
    the replay buffer.

    The `extract_batch` method is overloaded to retrieve additional information
    such as the `simulated_pathology`, `simulated_patients`, `simulated_severity`
    and eventually the `intermediate_states` in cases `n_step_return` > 1.
    """

    def __init__(
        self, *args, intermediate_data_flag=False, **kwargs,
    ):
        """Instantiates a class object.

        Parameters
        ----------
        intermediate_data_flag: bool
            Flag for extracting intermediate state data from replay buffer.
            default: False

        """
        super().__init__(*args, **kwargs)
        self._set_intermediate_data_flag(intermediate_data_flag)


class AugUniformReplayBuffer(AugReplayBufferExtractMixin, UniformReplayBuffer):
    """Overloads the `UniformReplayBuffer` class with custom functions.

    UniformReplayBuffer with custom functions dedicated to fetch data from
    the replay buffer.

    The `extract_batch` method is overloaded to retrieve additional information
    such as the `simulated_pathology`, `simulated_patients`, `simulated_severity`
    and eventually the `intermediate_states` in cases `n_step_return` > 1.
    """

    def __init__(
        self, *args, intermediate_data_flag=False, **kwargs,
    ):
        """Instantiates a class object.

        Parameters
        ----------
        intermediate_data_flag: bool
            Flag for extracting intermediate state data from replay buffer.
            default: False

        """
        super().__init__(*args, **kwargs)
        self._set_intermediate_data_flag(intermediate_data_flag)


class AugAsyncUniformReplayBuffer(
    AugReplayBufferExtractMixin, AsyncUniformReplayBuffer
):
    """Overloads the `AsyncUniformReplayBuffer` class with custom functions.

    AsyncUniformReplayBuffer with custom functions dedicated to fetch data from
    the replay buffer.

    The `extract_batch` method is overloaded to retrieve additional information
    such as the `simulated_pathology`, `simulated_patients`, `simulated_severity`
    and eventually the `intermediate_states` in cases `n_step_return` > 1.
    """

    def __init__(
        self, *args, intermediate_data_flag=False, **kwargs,
    ):
        """Instantiates a class object.

        Parameters
        ----------
        intermediate_data_flag: bool
            Flag for extracting intermediate state data from replay buffer.
            default: False

        """
        super().__init__(*args, **kwargs)
        self._set_intermediate_data_flag(intermediate_data_flag)


class AugUniformReplayFrameBuffer(
    AugReplayBufferExtractMixin, UniformReplayFrameBuffer
):
    """Overloads the `UniformReplayFrameBuffer` class with custom functions.

    UniformReplayFrameBuffer with custom functions dedicated to fetch data from
    the replay buffer.

    The `extract_batch` method is overloaded to retrieve additional information
    such as the `simulated_pathology`, `simulated_patients`, `simulated_severity`
    and eventually the `intermediate_states` in cases `n_step_return` > 1.
    """

    def __init__(
        self, *args, intermediate_data_flag=False, **kwargs,
    ):
        """Instantiates a class object.

        Parameters
        ----------
        intermediate_data_flag: bool
            Flag for extracting intermediate state data from replay buffer.
            default: False

        """
        super().__init__(*args, **kwargs)
        self._set_intermediate_data_flag(intermediate_data_flag)


class AugPrioritizedReplayFrameBuffer(
    AugReplayBufferExtractMixin, PrioritizedReplayFrameBuffer
):
    """Overloads the `PrioritizedReplayFrameBuffer` class with custom functions.

    PrioritizedReplayFrameBuffer with custom functions dedicated to
    fetch data from the replay buffer.

    The `extract_batch` method is overloaded to retrieve additional information
    such as the `simulated_pathology`, `simulated_patients`, `simulated_severity`
    and eventually the `intermediate_states` in cases `n_step_return` > 1.
    """

    def __init__(
        self, *args, intermediate_data_flag=False, **kwargs,
    ):
        """Instantiates a class object.

        Parameters
        ----------
        intermediate_data_flag: bool
            Flag for extracting intermediate state data from replay buffer.
            default: False

        """
        super().__init__(*args, **kwargs)
        self._set_intermediate_data_flag(intermediate_data_flag)


class AugAsyncUniformReplayFrameBuffer(
    AugReplayBufferExtractMixin, AsyncUniformReplayFrameBuffer
):
    """Overloads the `AsyncUniformReplayFrameBuffer` class with custom functions.

    AsyncUniformReplayFrameBuffer with custom functions dedicated
    to fetch data from the replay buffer.

    The `extract_batch` method is overloaded to retrieve additional information
    such as the `simulated_pathology`, `simulated_patients`, `simulated_severity`
    and eventually the `intermediate_states` in cases `n_step_return` > 1.
    """

    def __init__(
        self, *args, intermediate_data_flag=False, **kwargs,
    ):
        """Instantiates a class object.

        Parameters
        ----------
        intermediate_data_flag: bool
            Flag for extracting intermediate state data from replay buffer.
            default: False

        """
        super().__init__(*args, **kwargs)
        self._set_intermediate_data_flag(intermediate_data_flag)


class AugAsyncPrioritizedReplayFrameBuffer(
    AugReplayBufferExtractMixin, AsyncPrioritizedReplayFrameBuffer
):
    """Overloads the `AsyncPrioritizedReplayFrameBuffer` class with custom functions.

    AsyncPrioritizedReplayFrameBuffer with custom functions dedicated
    to fetch data from the replay buffer.

    The `extract_batch` method is overloaded to retrieve additional information
    such as the `simulated_pathology`, `simulated_patients`, `simulated_severity`
    and eventually the `intermediate_states` in cases `n_step_return` > 1.
    """

    def __init__(
        self, *args, intermediate_data_flag=False, **kwargs,
    ):
        """Instantiates a class object.

        Parameters
        ----------
        intermediate_data_flag: bool
            Flag for extracting intermediate state data from replay buffer.
            default: False

        """
        super().__init__(*args, **kwargs)
        self._set_intermediate_data_flag(intermediate_data_flag)


class AugUniformSequenceReplayFrameBuffer(
    AugSeqReplayBufferExtractMixin, UniformSequenceReplayFrameBuffer
):
    """Overloads the `UniformSequenceReplayFrameBuffer` class with custom functions.

    UniformSequenceReplayFrameBuffer with custom functions dedicated
    to fetch data from the replay buffer.

    The `extract_batch` method is overloaded to retrieve additional information
    such as the `simulated_pathology`, `simulated_patients`, `simulated_severity`.
    """

    pass


class AugPrioritizedSequenceReplayFrameBuffer(
    AugSeqReplayBufferExtractMixin, PrioritizedSequenceReplayFrameBuffer
):
    """Overloads the `PrioritizedSequenceReplayFrameBuffer` class with custom functions.

    PrioritizedSequenceReplayFrameBuffer with custom functions dedicated
    to fetch data from the replay buffer.

    The `extract_batch` method is overloaded to retrieve additional information
    such as the `simulated_pathology`, `simulated_patients`, `simulated_severity`.
    """

    pass


class AugAsyncUniformSequenceReplayFrameBuffer(
    AugSeqReplayBufferExtractMixin, AsyncUniformSequenceReplayFrameBuffer
):
    """Overload the `AsyncUniformSequenceReplayFrameBuffer` class with custom functions.

    AsyncUniformSequenceReplayFrameBuffer with custom functions dedicated
    to fetch data from the replay buffer.

    The `extract_batch` method is overloaded to retrieve additional information
    such as the `simulated_pathology`, the `simulated_patients`.
    """

    pass


class AugAsyncPrioritizedSequenceReplayFrameBuffer(
    AugSeqReplayBufferExtractMixin, AsyncPrioritizedSequenceReplayFrameBuffer
):
    """Overload `AsyncPrioritizedSequenceReplayFrameBuffer` class with custom functions.

    AsyncPrioritizedSequenceReplayFrameBuffer with custom functions dedicated
    to fetch data from the replay buffer.

    The `extract_batch` method is overloaded to retrieve additional information
    such as the `simulated_pathology`, `simulated_patients`, `simulated_severity`.
    """

    pass


class AugPrioritizedSequenceReplayBuffer(
    AugSeqReplayBufferExtractMixin, PrioritizedSequenceReplayBuffer
):
    """Overloads the `PrioritizedSequenceReplayBuffer` class with custom functions.

    PrioritizedSequenceReplayBuffer with custom functions dedicated to fetch data from
    the replay buffer.

    The `extract_batch` method is overloaded to retrieve additional information
    such as the `simulated_pathology`, `simulated_patients`, `simulated_severity`.
    """

    pass


class AugAsyncPrioritizedSequenceReplayBuffer(
    AugSeqReplayBufferExtractMixin, AsyncPrioritizedSequenceReplayBuffer
):
    """Overloads the `AsyncPrioritizedSequenceReplayBuffer` class with custom functions.

    AsyncPrioritizedSequenceReplayBuffer with custom functions dedicated to fetch
    data from the replay buffer.

    The `extract_batch` method is overloaded to retrieve additional information
    such as the `simulated_pathology`, `simulated_patients`, `simulated_severity`.
    """

    pass


class AugUniformSequenceReplayBuffer(
    AugSeqReplayBufferExtractMixin, UniformSequenceReplayBuffer
):
    """Overloads the `UniformSequenceReplayBuffer` class with custom functions.

    UniformSequenceReplayBuffer with custom functions dedicated to fetch data from
    the replay buffer.

    The `extract_batch` method is overloaded to retrieve additional information
    such as the `simulated_pathology`, `simulated_patients`, `simulated_severity`.
    """

    pass


class AugAsyncUniformSequenceReplayBuffer(
    AugSeqReplayBufferExtractMixin, AsyncUniformSequenceReplayBuffer
):
    """Overloads the `AsyncUniformSequenceReplayBuffer` class with custom functions.

    AsyncUniformSequenceReplayBuffer with custom functions dedicated to fetch data from
    the replay buffer.

    The `extract_batch` method is overloaded to retrieve additional information
    such as the `simulated_pathology`, `simulated_patients`, `simulated_severity`.
    """

    pass


class ReplayBufferFactory:
    """A factory for for instanciating replay buffer object.

    The allowed replay buffers are the ones defined in the rlpyt package.
    Please refer to this link to have an exhaustive list:
        https://rlpyt.readthedocs.io/en/latest/pages/replay.html

    All these replay buffers have been augmented to be able to store additional info
    useful for this project. The list of extra stored data is:
        - sim_patho: the simulated pathology
        - sim_patient: the simulated patient
        - sim_severity: the severity of the simulated pathology
        - sim_evidence: indicator of the patient experiencing the inquired action
        - sim_timestep: the timestep in the interaction session
        - sim_differential_indices: the indices of pathos within the differential
        - sim_differential_probas: the probas of pathos within the differential

    """

    def __init__(self):
        """Instantiates the factory.
        """
        async_pri_rep_frame = "AsyncPrioritizedReplayFrameBuffer".lower()
        async_pri_seq_rep_frame = "AsyncPrioritizedSequenceReplayFrameBuffer".lower()
        async_uni_seq_rep_frame = "AsyncUniformSequenceReplayFrameBuffer".lower()
        pri_seq_rep_frame = "PrioritizedSequenceReplayFrameBuffer".lower()
        async_pri_seq_rep = "AsyncPrioritizedSequenceReplayBuffer".lower()
        uni_seq_rep_frame = "UniformSequenceReplayFrameBuffer".lower()
        async_uni_seq_rep = "AsyncUniformSequenceReplayBuffer".lower()
        pri_seq_rep = "PrioritizedSequenceReplayBuffer".lower()
        uni_seq_rep = "UniformSequenceReplayBuffer".lower()
        self._builders = {
            async_pri_rep_frame: AugAsyncPrioritizedReplayFrameBuffer,
            "AsyncUniformReplayFrameBuffer".lower(): AugAsyncUniformReplayFrameBuffer,
            "PrioritizedReplayFrameBuffer".lower(): AugPrioritizedReplayFrameBuffer,
            "UniformReplayFrameBuffer".lower(): AugUniformReplayFrameBuffer,
            "AsyncPrioritizedReplayBuffer".lower(): AugAsyncPrioritizedReplayBuffer,
            "PrioritizedReplayBuffer".lower(): AugPrioritizedReplayBuffer,
            "AsyncTlPrioritizedReplayBuffer".lower(): AugAsyncTlPrioritizedReplayBuffer,
            "AsyncTlUniformReplayBuffer".lower(): AugAsyncTlUniformReplayBuffer,
            "TlPrioritizedReplayBuffer".lower(): AugTlPrioritizedReplayBuffer,
            "TlUniformReplayBuffer".lower(): AugTlUniformReplayBuffer,
            "AsyncUniformReplayBuffer".lower(): AugAsyncUniformReplayBuffer,
            "UniformReplayBuffer".lower(): AugUniformReplayBuffer,
            async_pri_seq_rep_frame: AugAsyncPrioritizedSequenceReplayFrameBuffer,
            async_uni_seq_rep_frame: AugAsyncUniformSequenceReplayFrameBuffer,
            pri_seq_rep_frame: AugPrioritizedSequenceReplayFrameBuffer,
            uni_seq_rep_frame: AugUniformSequenceReplayFrameBuffer,
            async_pri_seq_rep: AugAsyncPrioritizedSequenceReplayBuffer,
            pri_seq_rep: AugPrioritizedSequenceReplayBuffer,
            async_uni_seq_rep: AugAsyncUniformSequenceReplayBuffer,
            uni_seq_rep: AugUniformSequenceReplayBuffer,
        }

    def register_builder(self, key, builder, force_replace=False):
        """Registers an instance within the factory.

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

    def get_replay_buffer_class(self, key):
        """Get an agent class based on the provided key.

        Parameters
        ----------
        key: str
            key indication of which agent class to retrieve.

        Return
        ------
        cls: object
            class of the replay buffer.

        """
        builder = self._builders.get(key.lower())
        if not builder:
            raise ValueError(key + " is not a valid replay buffer key")
        return builder

    def create(self, key, *args, **kwargs):
        """Get a replay buffer instance based on the provided key.

        Parameters
        ----------
        key: str
            key indication of which replay buffer instance to create.
        args: list
            list of arguments.
        kwargs: dict
            dict of  arguments.

        Return
        ------
        result: object
            the instantiated replay buffer.

        """
        builder = self._builders.get(key.lower())
        if not builder:
            raise ValueError(key + " is not a valid replay buffer key")
        return builder(*args, **kwargs)
