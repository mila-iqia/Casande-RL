import torch.nn.functional as F
from rlpyt.utils.logging import logger


def binary_cross_entropy_loss(
    pred, target, weight=None, reduction="mean", **kwargs,
):
    """A binary cross entropy loss with generic inputs.

    Parameters
    ----------
    pred: tensor
        the predicted tensor.
    target: tensor
        the target tensor.
    weight: tensor
        a manual rescaling weight given to each class. Default: None
    reduction: str
        Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        Default: 'mean'

    Return
    ------
    result: tensor
        the computed loss.
    log_data: dict
        dictionary containing sub-component data to be logged.

    """
    result = F.binary_cross_entropy_with_logits(
        pred, target, weight=weight, reduction=reduction
    )
    return result, {}


def mse_loss(
    pred, target, weight=None, reduction="mean", **kwargs,
):
    """A MSE loss with generic inputs.

    Parameters
    ----------
    pred: tensor
        the predicted tensor.
    target: tensor
        the target tensor.
    weight: tensor
        a manual rescaling weight given to each class. Default: None
    reduction: str
        Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        Default: 'mean'

    Return
    ------
    result: tensor
        the computed loss.
    log_data: dict
        dictionary containing sub-component data to be logged.

    """
    result = F.mse_loss(pred, target, reduction=reduction)
    return result, {}


class RebuildLossFactory:
    """A factory for computing different feature rebuilding losses.

    The predefined classifier loss/reward functions are:
        - binary_cross_entropy
        - mse

    """

    def __init__(self):
        self._builders = {
            "bce": binary_cross_entropy_loss,
            "mse": mse_loss,
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

    def evaluate(
        self, key, pred, target, weight=None, reduction="mean", **kwargs,
    ):
        """Evaluates a loss (defined by the provided key) given the provided data.

        Parameters
        ----------
        key: str
            key indication of which loss to use for the evaluation.
        pred: tensor
            the predicted tensor.
        target: tensor
            the target tensor.
        weight: tensor
            a manual rescaling weight given to each class. Default: None
        reduction: str
            Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
            Default: 'mean'

        Return
        ------
        result: tensor
            the computed loss.
        log_data: dict
            dictionary containing sub-component data to be logged.

        """
        builder = self._builders.get(key.lower())
        if not builder:
            raise ValueError(key + " is not a valid classifier loss key")
        return builder(pred, target, weight=weight, reduction=reduction, **kwargs,)
