import numpy as np
import torch
import torch.nn.functional as F
from rlpyt.utils.logging import logger

from chloe.utils.scheduler_utils import sigmoid_scheduler
from chloe.utils.tensor_utils import (
    get_nb_pathos,
    get_nb_severe_pathos,
    soft_cross_entropy,
)


def cross_entropy_loss(
    pred,
    target,
    differential_indices,
    differential_probas,
    weight=None,
    reduction="mean",
    severity=None,
    timestep=None,
    **kwargs,
):
    """A cross entropy loss with generic inputs.

    Parameters
    ----------
    pred: tensor
        the predicted tensor.
    target: tensor
        the target tensor.
    differential_indices: tensor
        indices of the pathologies involved in the differential.
    differential_probas: tensor
        probabilities associated to the pathologies involved in the differential.
    weight: tensor
        a manual rescaling weight given to each class. Default: None
    reduction: str
        Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        Default: 'mean'
    severity: tensor
        the severity associated to each class. It is a 1-D tensor of size num_class.
        Default: None
    timestep: tensor
        the time step (within the interaction session) associated to each element
        of the batch. Default: None

    Return
    ------
    result: tensor
        the computed loss.
    log_data: dict
        dictionary containing sub-component data to be logged.

    """
    is_soft = (differential_indices is not None) and (differential_probas is not None)
    if is_soft:
        result = soft_cross_entropy(
            pred,
            differential_indices,
            differential_probas,
            weight=weight,
            reduction=reduction,
        )
    else:
        result = F.cross_entropy(pred, target, weight=weight, reduction=reduction)
    return result, {}


def sigmoid_modulated_cross_entropy_and_entropy_loss(
    pred,
    target,
    differential_indices,
    differential_probas,
    weight=None,
    reduction="mean",
    severity=None,
    timestep=None,
    ent_weight=0.0,
    alpha=50.0,
    use_severity_as_weight=False,
    **kwargs,
):
    """A sigmoid modulated [cross entropy + entropy] loss with generic inputs.

    Parameters
    ----------
    pred: tensor
        the predicted tensor.
    target: tensor
        the target tensor.
    differential_indices: tensor
        indices of the pathologies involved in the differential.
    differential_probas: tensor
        probabilities associated to the pathologies involved in the differential.
    weight: tensor
        a manual rescaling weight given to each class. Default: None.
    reduction: str
        Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        Default: 'mean'
    severity: tensor
        the severity associated to each class. It is a 1-D tensor of size num_class.
        Default: None
    timestep: tensor
        the time step (within the interaction session) associated to each element
        of the batch. Default: None
    ent_weight: float
        the weight of the entropy component. Default: 0.0
    alpha: float
        Skew parameter to shift the schedule for the sigmoid scheduler. Default: 50
    use_severity_as_weight: bool
        flag indicating whether or not the (inverse of the) pathology severity should
        be used as class weights when computing cross-entropy loss. Default: False
    max_turns: int
        the maximum number of steps allowed within an interaction session. Default: 30
    min_turns: int
        the minimum number of steps allowed within an interaction session. Default: 0
    min_map_val: float
        the minimum x value considered from the sigmoid function. Default: -10
    max_map_val: float
        the maxmum x value considered from the sigmoid function. Default: 10

    Return
    ------
    result: tensor
        the computed loss.
    log_data: dict
        dictionnary containing sub-component data to be logged.

    """
    log_dict = {}

    sig_weight = sigmoid_scheduler(timestep, alpha=alpha, **kwargs)
    entropy = (
        0.0
        if ent_weight == 0.0
        else -(F.softmax(pred, dim=1) * F.log_softmax(pred, dim=1)).sum(dim=1)
    )
    if (weight is None) and (severity is not None) and use_severity_as_weight:
        severity = severity.float()
        weight = 1.0 / (severity - severity.min() + 1)

    is_soft = (differential_indices is not None) and (differential_probas is not None)
    if is_soft:
        cross_entropy = soft_cross_entropy(
            pred,
            differential_indices,
            differential_probas,
            weight=weight,
            reduction="none",
        )
    else:
        cross_entropy = F.cross_entropy(pred, target, weight=weight, reduction="none")

    weighted_entropy = ent_weight * entropy
    modulated_weighted_entropy = sig_weight * weighted_entropy
    modulated_cross_entropy = sig_weight * cross_entropy

    if ent_weight != 0:
        log_dict["Entropy"] = weighted_entropy
        log_dict["ModulatedEntropy"] = modulated_weighted_entropy
    log_dict["CrossEntropy"] = cross_entropy
    log_dict["ModulatedCrossEntropy"] = modulated_cross_entropy

    loss = modulated_weighted_entropy + modulated_cross_entropy
    if reduction == "mean":
        loss = loss.mean()
        for k in log_dict.keys():
            log_dict[k] = log_dict[k].mean()
    elif reduction == "sum":
        loss = loss.sum()
        for k in log_dict.keys():
            log_dict[k] = log_dict[k].sum()

    return loss, log_dict


def sigmoid_modulated_cross_entropy_and_entropy_neg_reward(
    pred,
    target,
    differential_indices,
    differential_probas,
    weight=None,
    reduction="mean",
    severity=None,
    timestep=None,
    severity_threshold=3,
    diff_proba_threshold=0.01,
    ent_weight=1.0,
    ce_weight=1.0,
    sev_in_weight=0.0,
    sev_out_weight=0.0,
    sev_f1_weight=0.0,
    pat_in_weight=0.0,
    pat_out_weight=0.0,
    pat_f1_weight=0.0,
    initial_penalty=60.0,
    alpha=5,
    penalty_alpha=6,
    use_severity_as_weight=False,
    reverse_entropy_reward_flag=False,
    reverse_reward_flag=False,
    should_zero_centered_ce=False,
    **kwargs,
):
    """A sigmoid modulated [cross entropy + entropy] negative reward function.

    Parameters
    ----------
    pred: tensor
        the predicted tensor.
    target: tensor
        the target tensor.
    differential_indices: tensor
        indices of the pathologies involved in the differential.
    differential_probas: tensor
        probabilities associated to the pathologies involved in the differential.
    weight: tensor
        a manual rescaling weight given to each class. Default: None
    reduction: str
        Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        Default: 'mean'
    severity: tensor
        the severity associated to each class. It is a 1-D tensor of size num_class.
        Default: None
    timestep: tensor
        the time step (within the interaction session) associated to each element
        of the batch. Default: None
    severity_threshold : int
        threshold of the severity below which the pathology is considered severe.
        Default: 3
    diff_proba_threshold: float
        the threshold for a pathology to be part of the differential. Default: 0.01
    ent_weight: float
        the weight of the entropy component. Default: 1.0
    ce_weight: float
        the weight of the cross-entropy component. Default: 1.0
    sev_in_weight: float
        scaling factor of the number based components for ruling in severe pathologies.
        Default: 0
    sev_out_weight: float
        scaling factor of the number based components for ruling out severe pathologies.
        Default: 0
    sev_f1_weight: float
        scaling factor of the number based components for ruling severe pathologies.
        Default: 0
    pat_in_weight: float
        scaling factor of the number based components for ruling in pathologies.
        Default: 0
    pat_out_weight: float
        scaling factor of the number based components for ruling out pathologies.
        Default: 0
    pat_f1_weight: float
        scaling factor of the number based components for ruling pathologies.
        Default: 0
    initial_penalty: float
        penalty incurred if inferring at the beggining of the session.
        It is modulated by the scheduled_weight. So, it vanished over time.
        Default: 60.0
    alpha: float
        the delta value for translating the rescaled sigmoid function
        for the cross entropy reward component. Default: 5
    penalty_alpha: float
        the delta value for translating the rescaled sigmoid function
        for the initial penalty reward component. Default: 6
    use_severity_as_weight: bool
        flag indicating whether or not the (inverse of the) pathology severity should
        be used as class weights when computing cross-entropy loss. Default: False
    reverse_entropy_reward_flag: bool
        if True, instead of using the computed `entropy` (resp. `cross_entropy`) value,
        then `1 - entropy/max_entropy` (resp. `torch.exp(-ce_value)`) will be used
        instead. Default: False
    reverse_reward_flag: bool
        if True, instead of using the computed `ruleIn` (resp. `ruleOut` or `F1`) value,
        then `-ruleIn` (resp. `-ruleOut` or `-F1`) will be used
        instead. Default: False
    should_zero_centered_ce: bool
        if True, and reverse_entropy_reward_flag is True, then scale ce in [-1, 1].
        Default: False
    max_turns: int
        the maximum number of steps allowed within an interaction session. Default: 30
    min_turns: int
        the minimum number of steps allowed within an interaction session. Default: 0
    min_map_val: float
        the minimum x value considered from the sigmoid function. Default: -10
    max_map_val: float
        the maxmum x value considered from the sigmoid function. Default: 10

    Return
    ------
    result: tensor
        the computed loss.
    log_data: dict
        dictionnary containing sub-component data to be logged.
    """

    def log_data(data_dict, key, weight, value):
        if weight != 0:
            data_dict[key] = value
        return data_dict

    log_dict = {}
    ent_max_value = np.log(pred.size(1))

    # scheduler
    sig_weight = sigmoid_scheduler(timestep, alpha=alpha, **kwargs)
    pen_sig_weight = (
        0.0
        if initial_penalty == 0.0
        else (
            sig_weight
            if alpha == penalty_alpha
            else sigmoid_scheduler(timestep, alpha=penalty_alpha, **kwargs)
        )
    )

    # softmax
    sev_off = sev_out_weight == 0.0 and sev_in_weight == 0.0 and sev_f1_weight == 0.0
    sev_off = sev_off or severity is None or (severity < severity_threshold).sum() == 0
    pat_off = pat_out_weight == 0.0 and pat_in_weight == 0.0 and pat_f1_weight == 0.0
    p_softmax = (
        0.0 if (ent_weight == 0.0 and sev_off and pat_off) else F.softmax(pred, dim=1)
    )

    # entropy
    entropy = (
        0.0
        if ent_weight == 0.0
        else -(p_softmax * F.log_softmax(pred, dim=1)).sum(dim=1)
    )
    entropy = entropy / ent_max_value

    # cross entropy
    if (weight is None) and (severity is not None) and use_severity_as_weight:
        severity = severity.float()
        weight = 1.0 / (severity - severity.min() + 1)

    is_soft = (differential_indices is not None) and (differential_probas is not None)
    cross_entropy = (
        soft_cross_entropy(
            pred,
            differential_indices,
            differential_probas,
            weight=weight,
            reduction="none",
        )
        if is_soft
        else F.cross_entropy(pred, target, weight=weight, reduction="none")
    )
    maskProbs = (
        (differential_indices != -1).float() * differential_probas if is_soft else None
    )
    target_ent = (
       1
       if not is_soft
       else torch.exp(-((maskProbs * torch.log(maskProbs + 1e-18)).sum(dim=1)))
    )

    # severe pathology
    sev_f1 = 0.0
    sev_ruleOut, sev_ruleIn, sev_gt = (
        (0, 0, 0)
        if sev_off
        else get_nb_severe_pathos(
            p_softmax,
            target,
            differential_indices,
            differential_probas,
            severity,
            severity_threshold,
            diff_proba_threshold,
        )[1:4]
    )
    if not sev_off:
        max_sev = 0 if severity is None else (severity < severity_threshold).sum()
        one_val = 1 if severity is None else torch.ones_like(sev_gt)
        max_func = max if severity is None else torch.maximum
        sev_ruleOut = (max_sev - sev_gt - sev_ruleOut) / max_func(
            one_val, max_sev - sev_gt
        )
        sev_ruleIn = (sev_gt - sev_ruleIn) / max_func(one_val, sev_gt)
        sev_f1 = (2 * sev_ruleOut * sev_ruleIn) / (sev_ruleOut + sev_ruleIn + 1e-8)
        # the value are reversed because we are computing  a loss to minimize
        sev_ruleOut = 1.0 - sev_ruleOut if not reverse_reward_flag else -sev_ruleOut
        sev_ruleIn = 1.0 - sev_ruleIn if not reverse_reward_flag else -sev_ruleIn
        sev_f1 = 1.0 - sev_f1 if not reverse_reward_flag else -sev_f1

    # global pathology
    pat_f1 = 0.0
    pat_ruleOut, pat_ruleIn, pat_gt = (
        (0, 0, 0)
        if pat_off
        else get_nb_pathos(
            p_softmax,
            target,
            differential_indices,
            differential_probas,
            severity,
            severity_threshold,
            diff_proba_threshold,
            False,
        )[1:4]
    )
    if not pat_off:
        max_pat = pred.size(1)
        one_val = torch.ones_like(pat_gt)
        max_func = torch.maximum
        pat_ruleOut = (max_pat - pat_gt - pat_ruleOut) / max_func(
            one_val, max_pat - pat_gt
        )
        pat_ruleIn = (pat_gt - pat_ruleIn) / max_func(one_val, pat_gt)
        pat_f1 = (2 * pat_ruleOut * pat_ruleIn) / (pat_ruleOut + pat_ruleIn + 1e-8)
        # the value are reversed because we are computing  a loss to minimize
        pat_ruleOut = 1.0 - pat_ruleOut if not reverse_reward_flag else -pat_ruleOut
        pat_ruleIn = 1.0 - pat_ruleIn if not reverse_reward_flag else -pat_ruleIn
        pat_f1 = 1.0 - pat_f1 if not reverse_reward_flag else -pat_f1

    reversed_entropy = (
        entropy
        if (not reverse_entropy_reward_flag) or (ent_weight == 0)
        else -(1 - entropy)
    )

    reversed_cross_entropy = (
        cross_entropy
        if (not reverse_entropy_reward_flag)
        else (
            -torch.exp(-cross_entropy) * target_ent
            if not should_zero_centered_ce
            else 1.0 + 2 * (-torch.exp(-cross_entropy) * target_ent)
        )
    )
    modulated_penalty = initial_penalty * (1.0 - pen_sig_weight)
    weighted_entropy = ent_weight * reversed_entropy
    weighted_sev_out = sev_out_weight * sev_ruleOut
    weighted_sev_in = sev_in_weight * sev_ruleIn
    weighted_sev_f1 = sev_f1_weight * sev_f1
    weighted_pat_out = pat_out_weight * pat_ruleOut
    weighted_pat_in = pat_in_weight * pat_ruleIn
    weighted_pat_f1 = pat_f1_weight * pat_f1
    log_dict = log_data(log_dict, "Entropy", ent_weight, weighted_entropy)
    log_dict = log_data(log_dict, "SeverityRuleOut", sev_out_weight, weighted_sev_out)
    log_dict = log_data(log_dict, "SeverityRuleIn", sev_in_weight, weighted_sev_in)
    log_dict = log_data(log_dict, "SeverityRuleF1", sev_f1_weight, weighted_sev_f1)
    log_dict = log_data(log_dict, "PathologyRuleOut", pat_out_weight, weighted_pat_out)
    log_dict = log_data(log_dict, "PathologyRuleIn", pat_in_weight, weighted_pat_in)
    log_dict = log_data(log_dict, "PathologyRuleF1", pat_f1_weight, weighted_pat_f1)
    log_dict = log_data(log_dict, "InitialPenalty", initial_penalty, modulated_penalty)
    log_dict["CrossEntropy"] = ce_weight * cross_entropy
    log_dict = log_data(
        log_dict,
        "ReversedEntropy",
        reverse_entropy_reward_flag * ent_weight,
        reversed_entropy,
    )
    log_dict = log_data(
        log_dict,
        "ReversedCrossEntropy",
        reverse_entropy_reward_flag * ce_weight,
        reversed_cross_entropy,
    )

    loss = (
        modulated_penalty
        + weighted_entropy
        + (ce_weight * reversed_cross_entropy)
        + weighted_sev_out
        + weighted_sev_in
        + weighted_sev_f1
        + weighted_pat_out
        + weighted_pat_in
        + weighted_pat_f1
    )

    if reduction == "mean":
        loss = loss.mean()
        for k in log_dict.keys():
            log_dict[k] = log_dict[k].mean()
    elif reduction == "sum":
        loss = loss.sum()
        for k in log_dict.keys():
            log_dict[k] = log_dict[k].sum()

    return loss, log_dict


class ClassifierLossFactory:
    """A factory for computing different classifier losses/rewards.

    The predefined classifier loss/reward functions are:
        - cross_entropy
        - sigmoid_modulated_cross_entropy_and_entropy_loss
        - sigmoid_modulated_cross_entropy_and_entropy_neg_reward

    """

    def __init__(self):
        k1 = "sigmoid_modulated_cross_entropy_and_entropy_loss"
        k2 = "sigmoid_modulated_cross_entropy_and_entropy_neg_reward"
        self._builders = {
            "cross_entropy": cross_entropy_loss,
            k1: sigmoid_modulated_cross_entropy_and_entropy_loss,
            k2: sigmoid_modulated_cross_entropy_and_entropy_neg_reward,
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
        self,
        key,
        pred,
        target,
        differential_indices,
        differential_probas,
        weight=None,
        reduction="mean",
        severity=None,
        timestep=None,
        **kwargs,
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
        differential_indices: tensor
            indices of the pathologies involved in the differential.
        differential_probas: tensor
            probabilities associated to the pathologies involved in the differential.
        weight: tensor
            a manual rescaling weight given to each class. Default: None
        reduction: str
            Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
            Default: 'mean'
        severity: tensor
            the severity associated to each class. It is a 1-D tensor of size num_class.
            Default: None
        timestep: tensor
            the time step (within the interaction session) associated to each element
            of the batch. Default: None

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
        return builder(
            pred,
            target,
            differential_indices=differential_indices,
            differential_probas=differential_probas,
            weight=weight,
            reduction=reduction,
            severity=severity,
            timestep=timestep,
            **kwargs,
        )
