import torch


def sigmoid_scheduler(
    timestep,
    alpha=5,
    max_turns=30,
    min_turns=0,
    min_map_val=-10.0,
    max_map_val=10.0,
    is_decreasing=False,
    **kwargs
):
    """A weight scheduler based on the sigmoid function.

    This function copies the sigmoid function within the interval
    [`min_map_val`, `max_map_val`] and translates it horizontally
    in such a way that it starts at the point `0 - alpha`.

    Globally the implemented function is equal to:

    .. code-block:: text

        sigmoid(
         (((timestep - min_turns) * (max_map_val - min_map_val))/(max_turns- min_turns))
         + min_map_val + alpha
        ).


    Parameters
    ----------
    timestep: tensor
        the timesteps for which we want to compute the scheduling weight.
    alpha: float
        the delta value for translating the rescaled sigmoid function along the x-axis.
        Default: 5
    max_turns: int
        the number of maximum step allowed within an interaction session.
        Default: 30
    min_turns: int
        the number of minimum step allowed within an interaction session.
        Default: 0
    min_map_val: float
        the minimum x value considered from the sigmoid function. Default: -10
    max_map_val: float
        the maxmum x value considered from the sigmoid function. Default: 10
    is_decreasing: boolean
        Flag indicating whether the schedule is increasing or decreasing.
        Default: False

    Return
    ------
    result: tensor
        the computed scheduled weight.

    """
    if timestep is None:
        return 1.0
    x = (
        (
            ((timestep - min_turns) * (max_map_val - min_map_val))
            / max(1, (max_turns - min_turns))
        )
        + min_map_val
        + alpha
    )
    if is_decreasing:
        x *= -1.0
    return torch.sigmoid(x)


def bell_scheduler(
    timestep,
    alpha_a=5,
    alpha_b=None,
    max_turns=30,
    min_turns=0,
    min_map_val=-10.0,
    max_map_val=10.0,
    **kwargs
):
    """A weight scheduler based on the bell shape obtained by combination of 2 sigmoids.

    This function copies two sigmoid functions within the interval
    [`min_map_val`, `max_map_val`] and translates them horizontally
    in such a way that it starts at the point `0 - alpha_a`
    and `0 + alpha_b` respectively and then takes their difference.

    Globally the implemented function is equal to:

    .. code-block:: text

        sigmoid(
         (((timestep - min_turns) * (max_map_val - min_map_val))/(max_turns- min_turns))
         + min_map_val + alpha_a
        )
        - sigmoid(
         (((timestep - min_turns) * (max_map_val - min_map_val))/(max_turns- min_turns))
         + min_map_val - alpha_b
        ).


    Parameters
    ----------
    timestep: tensor
        the timesteps for which we want to compute the scheduling weight.
    alpha_a: float
        the delta value for translating the first rescaled sigmoid function
        along the x-axis. Default: 5
    alpha_b: float
        the delta value for translating the second rescaled sigmoid function
        along the x-axis. if None, it will take the same value as `alpha_a`.
        Default: None
    max_turns: int
        the number of maximum step allowed within an interaction session.
        Default: 30
    min_turns: int
        the number of minimum step allowed within an interaction session.
        Default: 0
    min_map_val: float
        the minimum x value considered from the sigmoid function. Default: -10
    max_map_val: float
        the maxmum x value considered from the sigmoid function. Default: 10

    Return
    ------
    result: tensor
        the computed scheduled weight.

    """
    if timestep is None:
        return 1.0
    if alpha_b is None:
        alpha_b = alpha_a
    x = (
        ((timestep - min_turns) * (max_map_val - min_map_val))
        / max(1, (max_turns - min_turns))
    ) + min_map_val
    return torch.sigmoid(x + alpha_a) - torch.sigmoid(x - alpha_b)
