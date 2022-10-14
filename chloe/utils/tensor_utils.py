import numpy as np
import torch


def _get_target_categorical_distributional(target, pi_src, V_min, V_max, lin_z=None):
    """Gets a categorical distribution of the target value given the init distribution.

    This function is useful in Categorical DQN Algorithm.

    Parameters
    ----------
    target: tensor
        the desired target value.
    pi_src: tensor
        the initial distribution.
    V_min: float
        the authorized min value of the `target` parameter.
    V_max: float
        the authorized max value of the `target` parameter.
    lin_z: tensor
        one-dimensional tensor whose values are evenly spaced
        from `V_min` to `V_max`, inclusive. Default: None

    Return
    ---------
    target_p: tensor
        the computed categorical distribution.

    """
    assert V_max is not None
    assert V_min is not None
    assert V_max > V_min
    n_atoms = pi_src.size(-1)
    assert n_atoms > 1
    delta_z = (V_max - V_min) / (n_atoms - 1)
    assert (lin_z is None) or (lin_z.ndim == 1 and lin_z.size(0) == n_atoms)
    tmp_lin_z = (
        torch.linspace(V_min, V_max, n_atoms, device=pi_src.device)
        if lin_z is None
        else lin_z
    )

    # clamp the target q values (B')
    target_values = torch.clamp(target, V_min, V_max)
    # expand to B'x P
    tmp_dim = len(target_values.size())
    target_values = target_values.unsqueeze(-1).expand(*([-1] * tmp_dim + [n_atoms]))

    z_bc = tmp_lin_z.view(*([1] * tmp_dim), -1, 1)  # [1,P,1]
    next_z_bc = target_values.unsqueeze(-2)  # [B',1,P']
    abs_diff_on_delta = abs(next_z_bc - z_bc) / delta_z
    projection_coeffs = torch.clamp(1 - abs_diff_on_delta, 0, 1)
    # projection_coeffs is a 3-D tensor: [B',P,P']
    # dim-0, dim-1: independent data entries
    # dim-1: base_z atoms (remains after projection)
    # dim-2: next_z atoms (summed in projection)

    # get the target_p
    target_p_unproj = pi_src  # [B',P']
    target_p_unproj = target_p_unproj.unsqueeze(-2)  # [B',1,P']
    target_p = (target_p_unproj * projection_coeffs).sum(-1)  # [B',P]

    return target_p


def _clamp_utils(value, min_val=None, max_val=None):
    """Clamps the provided values to be within the provided range.

    Parameters
    ----------
    value: tensor
        the value to be clamped.
    min_val: float
        the minimum value. Default: None
    max_val: float
        the maximum value. Default: None

    Return
    ---------
    result: tensor
        the clamped values.

    """
    if value is None:
        return value
    if min_val is not None:
        value = torch.clamp(value, min=min_val)
    if max_val is not None:
        value = torch.clamp(value, max=max_val)
    return value


def _negate_tensor(data):
    """Negates the value of the provided data

    Parameters
    ----------
    data: object
        the data whose values will be negated.

    Return
    ----------
    result: object
        the corresponding data with negated values.

    """
    if data is None:
        return None
    if isinstance(data, (float, np.ndarray, torch.Tensor)):
        return -data
    if isinstance(data, dict):
        return data.__class__({k: _negate_tensor(data[k]) for k in data.keys()})
    if isinstance(data, (tuple, list)):
        return data.__class__([_negate_tensor(k) for k in data])
    raise ValueError("Unknown type")


def get_nb_severe_pathos(
    predicted_differential,
    targets,
    differential_indices,
    differential_probas,
    severity=None,
    severity_threshold=3,
    diff_proba_threshold=0.01,
    **kwargs,
):
    """Get the number of servere pathologies between GT and the predicted differential.
    It returns:
        - the number of severe pathologies common to GT and Pred
        - the number of severe pathologies in Pred but not in GT
        - the number of severe pathologies in GT but not in Pred
        - the number of severe pathologies in GT
        - the number of severe pathologies in Pred

    Parameters
    ----------
    predicted_differential: tensor
        softmax values characterizing the predicted differential.
    targets: tensor
        targets associated to the end goal of the trajectory (predicted pathology).
    differential_indices: tensor
        indices of the pathologies involved in the differential.
    differential_probas: tensor
        probabilities associated to the pathologies involved in the differential.
    severity: tensor
        the severity associated to each class. It is a 1-D tensor of size num_class.
        Default: None
    severity_threshold : int
        threshold of the severity below which the pathology is considered severe.
        Default: 3
    diff_proba_threshold: float
        the threshold for a pathology to be part of the differential. Default: 0.01

    Return
    ------
    result: tuple of 5 tensors
        the number of severe pathologies as in the description.
    """
    return get_nb_pathos(
        predicted_differential,
        targets,
        differential_indices,
        differential_probas,
        severity,
        severity_threshold,
        diff_proba_threshold,
        True,
        **kwargs,
    )


def get_nb_pathos(
    predicted_differential,
    targets,
    differential_indices,
    differential_probas,
    severity=None,
    severity_threshold=3,
    diff_proba_threshold=0.01,
    severe_flag=True,
    **kwargs,
):
    """Get the number of (severe) pathologies between GT and the predicted differential.
    It returns:
        - the number of (severe) pathologies common to GT and Pred
        - the number of (severe) pathologies in Pred but not in GT
        - the number of (severe) pathologies in GT but not in Pred
        - the number of (severe) pathologies in GT
        - the number of (severe) pathologies in Pred

    Parameters
    ----------
    predicted_differential: tensor
        softmax values characterizing the predicted differential.
    targets: tensor
        targets associated to the end goal of the trajectory (predicted pathology).
    differential_indices: tensor
        indices of the pathologies involved in the differential.
    differential_probas: tensor
        probabilities associated to the pathologies involved in the differential.
    severity: tensor
        the severity associated to each class. It is a 1-D tensor of size num_class.
        Default: None
    severity_threshold : int
        threshold of the severity below which the pathology is considered severe.
        Default: 3
    diff_proba_threshold: float
        the threshold for a pathology to be part of the differential. Default: 0.01
    severe_flag: boolean
        flag indicating if the computed stats is for severe pathology only.
        Default: True

    Return
    ------
    result: tuple of 5 tensors
        the number of (severe) pathologies as in the description.
    """
    # we dont have information on severe pathologies
    # we return 0 for all the stats
    if severity is None and severe_flag:
        sh = list(predicted_differential.shape)
        sh = sh[0:1] + sh[2:]
        td = {
            "device": predicted_differential.device,
            "dtype": predicted_differential.dtype,
        }
        tz = torch.zeros
        return tz(*sh, **td), tz(*sh, **td), tz(*sh, **td), tz(*sh, **td), tz(*sh, **td)
    n = predicted_differential.shape[1]
    device = predicted_differential.device
    if (differential_indices is not None) and (differential_probas is not None):
        dtype = predicted_differential.dtype
        tmp_shape = list(predicted_differential.shape)
        tmp_shape[1] += 1
        gt = torch.zeros(*tmp_shape, dtype=dtype, device=device)
        tmp_mask = differential_indices != -1
        masked_ind = tmp_mask * differential_indices + torch.logical_not(tmp_mask) * n
        gt.scatter_(1, masked_ind, differential_probas)
        gt = gt[:, 0:-1]
    else:
        gt = torch.zeros_like(predicted_differential)
        gt.scatter_(1, targets.unsqueeze(1), 1)

    severity_mask = None
    if severe_flag:
        severity_mask = severity < severity_threshold  # severe pathologies
    else:
        # we consider all pathos
        severity_mask = torch.ones(n, device=device, dtype=torch.bool)

    if len(gt.shape) > 1:
        tmp_shape = [1] * len(gt.shape)
        tmp_shape[1] = -1
        severity_mask = severity_mask.view(*tmp_shape)

    # mask of severes patho in gt and pred
    gt_severity_mask = ((gt > diff_proba_threshold) * severity_mask).bool()
    pr_severity_mask = (
        (predicted_differential > diff_proba_threshold) * severity_mask
    ).bool()

    # mask of severe patho in pred and in gt
    pr_gt_common_severity_mask = (gt_severity_mask * pr_severity_mask).bool()

    # mask of severe patho in pred not in gt
    pr_gt_nonshared_severity_mask = (
        torch.logical_not(gt_severity_mask) * pr_severity_mask
    ).bool()

    # mask of severe patho in gt not in pred
    gt_pr_nonshared_severity_mask = (
        torch.logical_not(pr_severity_mask) * gt_severity_mask
    ).bool()

    return (
        pr_gt_common_severity_mask.long().sum(1),
        pr_gt_nonshared_severity_mask.long().sum(1),
        gt_pr_nonshared_severity_mask.long().sum(1),
        gt_severity_mask.long().sum(1),
        pr_severity_mask.long().sum(1),
    )


def soft_cross_entropy(
    pred, target_indices, target_probas, weight=None, reduction="mean", ignore_index=-1
):
    """Computes the cross entropy loss using soft labels.

    Here the soft labels are defined through the parameters `target_indices`
    and `target_probas`. They respectively represent the class indices involved
    in the target distribution and their corresponding probability.
    The provided `ignore_index` can be used as padding element in the `target_indices`
    field.

    Per definition, we have (https://en.wikipedia.org/wiki/Cross_entropy):
        CE(p,q) = -(p * log(q)).sum()
    With a provided weight per class, the computation becomes:
        CE(p,q,w) = -(p * log(q)).sum() * (p * w).sum()

    Parameters
    ----------
    pred: tensor
        a tensor of size `N x C x *` where N is the batch size, C is the number
        of classes, and `*` represents any other dimensions. This tensor represents
        the logit values.
    target_indices: tensor
        a tensor of size `N x D x *` where N is the batch size, D <= C is the number
        of classes present in the soft distribution, and `*` represents
        any other dimensions. It must match the tailing dimensions of `pred`.
    target_probas: tensor
        a tensor of same size as `target_indices` representing the probability
        associated to each class therein.
    weight: tensor
        a manual rescaling weight given to each class. It is a 1-D tensor of size
        `C`. Default: None
    reduction: str
        Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        Default: 'mean'
    ignore_index: int
        Specifies a target value that is ignored and does not contribute
        to the gradient. Default: -1

    Return
    ----------
    result: tensor
        the computed loss.

    """
    assert reduction in ["none", "mean", "sum"]

    dim = pred.dim()
    if dim < 2:
        raise ValueError("Expected 2 or more dimensions (got {})".format(dim))

    dim = target_indices.dim()
    if dim < 2:
        raise ValueError("Expected 2 or more dimensions (got {})".format(dim))

    assert (weight is None) or (weight.dim() == 1 and weight.size(0) == pred.size(1))

    if pred.size(0) != target_indices.size(0):
        raise ValueError(
            f"Expected input batch_size ({pred.size(0)}) to match "
            f"target batch_size ({target_indices.size(0)})."
        )

    if pred.size(1) < target_indices.size(1):
        raise ValueError(
            f"Expected input class_size ({pred.size(1)}) to be greater/equal"
            f"than target class_size ({target_indices.size(1)})."
        )
    if target_indices.size()[2:] != pred.size()[2:]:
        out_size = target_indices.size()[:2] + pred.size()[2:]
        raise ValueError(
            f"Expected target_indices size {out_size} (got {target_indices.size()})"
        )
    if target_indices.size() != target_probas.size():
        raise ValueError(
            f"Expected target_probas size {target_indices.size()} "
            f"(got {target_probas.size()})"
        )

    log_probs = torch.nn.functional.log_softmax(pred, dim=1)
    mask = target_indices != ignore_index
    masked_indices = target_indices * mask
    tmp_weight = 1.0 if weight is None else weight[masked_indices]
    avg_log_probs = (mask * log_probs.gather(1, masked_indices) * target_probas).sum(
        dim=1
    )
    avg_weight = (
        1.0 if weight is None else (tmp_weight * mask * target_probas).sum(dim=1)
    )
    result = -(avg_weight * avg_log_probs)

    if reduction == "sum":
        result = result.sum()
    elif reduction == "mean":
        result = result.mean() if weight is None else result.sum() / avg_weight.sum()

    return result
