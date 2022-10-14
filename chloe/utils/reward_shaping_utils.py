import math

import torch
import torch.nn.functional as F
from rlpyt.utils.logging import logger

from chloe.utils.scheduler_utils import bell_scheduler, sigmoid_scheduler
from chloe.utils.tensor_utils import (
    _clamp_utils,
    get_nb_pathos,
    get_nb_severe_pathos,
    soft_cross_entropy,
)

MAX_JS_BOUND = math.log(2)


def cross_entropy_reshaping(
    next_values,
    prev_values,
    targets,
    differential_indices,
    differential_probas,
    evidence,
    discount=None,
    severity=None,
    timestep=None,
    **kwargs,
):
    """A reward shaping function based on cross-entropy score.

    This  reward shaping function is of the form of:

    .. code-block:: text

        discount * f(next_values, targets) - f(prev_values, targets)

    where:
        - next_values is the result of some function ``g`` applied to state ``S_{t+1}``
        - prev_values is the result of the same function ``g`` applied to\
        state ``S_{t}``.

    Here, the function ``f`` is the (negative) cross-entropy function.

    Parameters
    ----------
    next_values: tensor
        logit values characterizing the distribution at state S_{t+1}.
    prev_values: tensor
        logit values characterizing the distribution at state S_{t}.
    targets: tensor
        targets associated to the end goal of the trajectory (predicted pathology).
    differential_indices: tensor
        indices of the pathologies involved in the differential.
    differential_probas: tensor
        probabilities associated to the pathologies involved in the differential.
    evidence: tensor
        indicator of the simulated patient experiencing the inquired action.
    discount: float
        the discount factor. Default: None
    severity: tensor
        the severity associated to each class. It is a 1-D tensor of size num_class.
        Default: None
    timestep: tensor
        the time step (within the interaction session) associated to each element
        of the batch. Default: None

    Return
    ------
    result: tensor
        the computed auxiliary reward.
    log_data: dict
        dictionnary containing sub-component data to be logged.

    """
    if discount is None:
        discount = 1.0

    is_soft = (differential_indices is not None) and (differential_probas is not None)

    next_values = next_values.transpose(1, -1)
    prev_values = prev_values.transpose(1, -1)

    if is_soft:
        differential_indices = differential_indices.transpose(1, -1)
        differential_probas = differential_probas.transpose(1, -1)

        next_metrics = -soft_cross_entropy(
            next_values, differential_indices, differential_probas, reduction="none"
        )
        prev_metrics = -soft_cross_entropy(
            prev_values, differential_indices, differential_probas, reduction="none"
        )
    else:
        next_metrics = -F.cross_entropy(next_values, targets, reduction="none")
        prev_metrics = -F.cross_entropy(prev_values, targets, reduction="none")

    return (discount * next_metrics) - prev_metrics, {}


def entropy_reshaping(
    next_values,
    prev_values,
    targets,
    differential_indices,
    differential_probas,
    evidence,
    discount=None,
    severity=None,
    timestep=None,
    **kwargs,
):
    """A reward shaping function based on entropy score.

    This  reward shaping function is of the form of:

    .. code-block:: text

        discount * f(next_values, targets) - f(prev_values, targets)

    where:
        - next_values is the result of some function ``g`` applied to state ``S_{t+1}``
        - prev_values is the result of the same function ``g`` applied to\
        state ``S_{t}``.

    Here, the function ``f`` is the (negative) entropy function.

    Parameters
    ----------
    next_values: tensor
        logit values characterizing the distribution at state S_{t+1}.
    prev_values: tensor
        logit values characterizing the distribution at the state S_{t}.
    targets: tensor
        targets associated to the end goal of the trajectory (predicted pathology).
    differential_indices: tensor
        indices of the pathologies involved in the differential.
    differential_probas: tensor
        probabilities associated to the pathologies involved in the differential.
    evidence: tensor
        indicator of the simulated patient experiencing the inquired action.
    discount: float
        the discount factor. Default: None
    severity: tensor
        the severity associated to each class. It is a 1-D tensor of size num_class.
        Default: None
    timestep: tensor
        the time step (within the interaction session) associated to each element
        of the batch. Default: None

    Return
    ------
    result: tensor
        the computed auxiliary reward.
    log_data: dict
        dictionnary containing sub-component data to be logged.

    """
    if discount is None:
        discount = 1.0

    next_metrics = F.softmax(next_values, dim=-1) * F.log_softmax(next_values, dim=-1)
    next_metrics = next_metrics.sum(dim=-1)
    prev_metrics = F.softmax(prev_values, dim=-1) * F.log_softmax(prev_values, dim=-1)
    prev_metrics = prev_metrics.sum(dim=-1)
    return (discount * next_metrics) - prev_metrics, {}


def ce_ent_sent_reshaping(
    next_values,
    prev_values,
    targets,
    differential_indices,
    differential_probas,
    evidence,
    discount=None,
    severity=None,
    timestep=None,
    severity_threshold=3,
    diff_proba_threshold=0.01,
    ce_alpha=5,
    ent_alpha=5,
    js_alpha=5,
    tv_alpha=5,
    pat_in_alpha=5,
    pat_out_alpha=5,
    pat_f1_alpha=5,
    sev_in_alpha=5,
    sev_out_alpha=5,
    sev_f1_alpha=5,
    sev_ent_alpha=8,
    sev_ent_alpha_b=1,
    ent_weight=1.0,
    ce_weight=1.0,
    js_weight=0.0,
    tv_weight=0.0,
    pat_in_weight=0.0,
    pat_out_weight=0.0,
    pat_f1_weight=0.0,
    sev_in_weight=0.0,
    sev_out_weight=0.0,
    sev_f1_weight=0.0,
    sev_ent_weight=1.0,
    sev_js_weight=0.0,
    sev_tv_weight=0.0,
    use_severity_as_weight=False,
    link_div_with_negative_evidence=False,
    normalize_sev_dist=False,
    reverse_ce_flag=False,
    reverse_flag=False,
    bounds_dict={},
    **kwargs,
):
    """A reward shaping function based on entropy, cross entopy, tv and JS scores.

    Specifically, it returns a weighted combnation of all those reward functions
    and their corresponding schedules as discussed here: shorturl.at/cvU56.

    This  reward shaping function is of the form of:

    .. code-block:: text

        discount * f(next_values, targets) - f(prev_values, targets)

    where:
        - next_values is the result of some function ``g`` applied to state ``S_{t+1}``
        - prev_values is the result of the same function ``g`` applied to\
        state ``S_{t}``.

    Here, the function ``f`` is a combination of:
        - entropy
        - cross-entropy (CE)
        - total variation (TV)
        - Jensen-Shannon divergence (JS).

    Parameters
    ----------
    next_values: tensor
        logit values characterizing the distribution at state S_{t+1}.
    prev_values: tensor
        logit values characterizing the distribution at the state S_{t}.
    targets: tensor
        targets associated to the end goal of the trajectory (predicted pathology).
    differential_indices: tensor
        indices of the pathologies involved in the differential.
    differential_probas: tensor
        probabilities associated to the pathologies involved in the differential.
    evidence: tensor
        indicator of the simulated patient experiencing the inquired action.
    discount: float
        the discount factor. Default: None
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
    ce_alpha: float
        skew parameter to shift the schedule for the CE sigmoid scheduler. Default: 5
    ent_alpha: float
        skew parameter to shift the schedule for the Entropy sigmoid scheduler.
        Default: 5
    js_alpha: float
        skew parameter to shift the schedule for the JS sigmoid scheduler. Default: 5
    tv_alpha: float
        skew parameter to shift the schedule for the TV sigmoid scheduler. Default: 5
    pat_in_alpha: float
        skew parameter to shift the schedule for the not yet recevered pathology
        sigmoid scheduler. Default: 5
    pat_out_alpha: float
        skew parameter to shift the schedule for the pathology exploration
        sigmoid scheduler. Default: 5
    pat_f1_alpha: float
        skew parameter to shift the schedule for the f1 pathology exploration
        sigmoid scheduler. Default: 5
    sev_in_alpha: float
        skew parameter to shift the schedule for the severe pathology ruling in
        sigmoid scheduler. Default: 5
    sev_out_alpha: float
        skew parameter to shift the schedule for the severe pathology ruling out
        sigmoid scheduler. Default: 5
    sev_f1_alpha: float
        skew parameter to shift the schedule for the f1 severe pathology
        sigmoid scheduler. Default: 5
    sev_ent_alpha: float
        skew parameter to shift the first sigmoid for the severe pathologies
        bell scheduler. Default: 8
    sev_ent_alpha_b: float
        skew parameter to shift the second sigmoid for the severe pathologies
        bell scheduler. Default: 1
    ent_weight: float
        scaling factor of the entropy based components. Default: 1
    ce_weight: float
        scaling factor of the CE based components. Default: 1
    js_weight: float
        scaling factor of the JS based components. Default: 0
    tv_weight: float
        scaling factor of the TV based components. Default: 0
    pat_in_weight: float
        scaling factor of the reward components for not yet recevered pathologies.
        Default: 0
    pat_out_weight: float
        scaling factor of the reward components for exploring no grounded pathologies.
        Default: 0
    pat_f1_weight: float
        scaling factor of the reward components for f1 based exploration of pathologies.
        Default: 0
    sev_in_weight: float
        scaling factor of the number based components for ruling in severe pathologies.
        Default: 0
    sev_out_weight: float
        scaling factor of the number based components for ruling out severe pathologies.
        Default: 0
    sev_f1_weight: float
        scaling factor of the number based components for in/out severe pathologies.
        Default: 0
    sev_ent_weight: float
        scaling factor of the entropy based components for severe pathologies.
        Default: 1
    sev_js_weight: float
        scaling factor of the JS based components for severe pathologies. Default: 0
    sev_tv_weight: float
        scaling factor of the TV based components for severe pathologies. Default: 0
    use_severity_as_weight: bool
        flag indicating whether or not the (inverse of the) pathology severity should
        be used as class weights when computing cross-entropy loss. Default: True
    link_div_with_negative_evidence: bool
        flag to indicate whether or not to consider divergence based reward componenents
        (JS, TV) only when evidence is no, i.e, the answer from the patient to a
        question is `no`. Default: False
    normalize_sev_dist: bool
        flag to indicate whether or not to re-normalize distribution on severe
        pathologies when computing related reward components. Default: False
    reverse_ce_flag: bool
        if True, instead of using the computed`cross_entropy` value,
        then `- exp(- ce_value)` will be used
        instead. Default: False
    reverse_flag: bool
        if True, instead of using the computed patho stats value,
        then `-(1 - value)
        instead. Default: False
    bounds_dict: dict
        dictionnary used to define the bounds of each reward components. The following
        keys can be expected:
            - ent_min, ent_max: bounds of the entropy based reward component
            - ce_min, ce_max: bounds of the CE based reward component
            - js_min, js_max: bounds of the JS based reward component
            - tv_min, tv_max: bounds of the TV based reward component
            - pat_out_min, pat_out_max: bounds of the pathologies exploration
              based reward component
            - pat_in_min, pat_in_max: bounds of the not yet recoved pathologies
              based reward component
            - pat_f1_min, pat_f1_max: bounds of the f1 recoved pathologies
              based reward component
            - sev_out_min, sev_out_max: bounds of the ruling out based reward component
              for severe pathologies
            - sev_in_min, sev_in_max: bounds of the ruling in based reward component
              for severe pathologies
            - sev_f1_min, sev_f1_max: bounds of the ruling out/in based reward component
              for severe pathologies
            - sev_ent_min, sev_ent_max: bounds of the entropy based reward component
              for severe pathologies
            - sev_js_min, sev_js_max: bounds of the JS based reward component for severe
              pathologies
            - sev_tv_min, sev_tv_max: bounds of the TV based reward component for severe
              pathologies
        Default: {}
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
        the computed auxiliary reward.
    log_data: dict
        dictionnary containing sub-component data to be logged.
    """

    def log_data(data_dict, key, weight, value, mod_value, raw_value, clamp_value):
        if weight != 0:
            data_dict[key] = value
            data_dict[f"Modulated{key}"] = mod_value
            data_dict[f"Raw{key}"] = raw_value
            data_dict[f"Clamp{key}"] = clamp_value
        return data_dict

    def total_variation(next_p, prev_p):
        return (
            0.0
            if next_p is None or prev_p is None
            else 0.5 * torch.abs(next_p - prev_p).sum(dim=1)
        )

    def kl_div(src_log_p, tgt_p):
        return (
            0.0
            if src_log_p is None or tgt_p is None
            else F.kl_div(src_log_p, tgt_p, reduction="none").sum(dim=1)
        )

    def js_div(src_p, tgt_p):
        m = (src_p + tgt_p) / 2.0
        log_m = torch.log(m + 1e-12)
        result = 0.5 * kl_div(log_m, src_p) + 0.5 * kl_div(log_m, tgt_p)
        result[result > MAX_JS_BOUND] = MAX_JS_BOUND
        return result

    def f(logits, is_soft, tgts, diff_indices, diff_probas):
        # entropy
        p = F.softmax(logits, dim=1)
        log_p = F.log_softmax(logits, dim=1)
        p_log_p = p * log_p
        entropy = -p_log_p.sum(dim=1)

        # cross entropy
        weight = None
        if severity is not None and use_severity_as_weight:
            tmp_severity = severity.float()
            weight = 1.0 / (tmp_severity - tmp_severity.min() + 1)
        cross_entropy = (
            soft_cross_entropy(
                logits, diff_indices, diff_probas, weight=weight, reduction="none"
            )
            if is_soft
            else F.cross_entropy(logits, tgts, weight=weight, reduction="none")
        )

        # entropy on severe pathology
        entropy_severe = 0.0
        p_severe = None
        log_p_severe = None
        if severity is not None:
            mask = severity < severity_threshold  # severe pathology
            sev_lgit = logits[:, mask]
            p_severe = F.softmax(sev_lgit, dim=1) if normalize_sev_dist else p[:, mask]
            log_p_severe = (
                F.log_softmax(sev_lgit, dim=1) if normalize_sev_dist else log_p[:, mask]
            )
            p_log_p_severe = p_severe * log_p_severe
            entropy_severe = -p_log_p_severe.sum(dim=1)

        # number of severe pathologies info
        sev_common, sev_predNoGT, sev_gtNoPred, sev_gt, sev_pred = get_nb_severe_pathos(
            p,
            tgts,
            diff_indices,
            diff_probas,
            severity,
            severity_threshold,
            diff_proba_threshold,
        )

        # number of pathologies info
        pat_common, pat_predNoGT, pat_gtNoPred, pat_gt, pat_pred = get_nb_pathos(
            p,
            tgts,
            diff_indices,
            diff_probas,
            severity,
            severity_threshold,
            diff_proba_threshold,
            False,
        )

        # combination
        result = (
            entropy,
            cross_entropy,
            entropy_severe,
            p_severe,
            log_p_severe,
            p,
            log_p,
            sev_common,
            sev_predNoGT,
            sev_gtNoPred,
            sev_gt,
            sev_pred,
            pat_common,
            pat_predNoGT,
            pat_gtNoPred,
            pat_gt,
            pat_pred,
        )
        return result

    if discount is None:
        discount = 1.0

    is_soft = (differential_indices is not None) and (differential_probas is not None)

    log_dict = {}

    next_values = next_values.transpose(1, -1)
    prev_values = prev_values.transpose(1, -1)

    differential_indices = (
        None if differential_indices is None else differential_indices.transpose(1, -1)
    )
    differential_probas = (
        None if differential_probas is None else differential_probas.transpose(1, -1)
    )

    (
        next_ent,
        next_ce,
        next_sev_ent,
        next_p_sev,
        _,
        next_p,
        _,
        _,
        next_nbSevPrNoGT,
        next_nbSevGTNoPr,
        gt_nbSev,
        _,
        _,
        next_nbPatPrNoGT,
        next_nbPatGTNoPr,
        gt_nbPat,
        _,
    ) = f(next_values, is_soft, targets, differential_indices, differential_probas)
    (
        prev_ent,
        prev_ce,
        prev_sev_ent,
        prev_p_sev,
        _,
        prev_p,
        _,
        _,
        prev_nbSevPrNoGT,
        prev_nbSevGTNoPr,
        _,
        _,
        _,
        prev_nbPatPrNoGT,
        prev_nbPatGTNoPr,
        _,
        _,
    ) = f(prev_values, is_soft, targets, differential_indices, differential_probas)

    # reverse number to take fraction
    max_sev = 0 if severity is None else (severity < severity_threshold).sum()
    one_val = torch.ones_like(gt_nbSev)
    max_func = torch.maximum
    sevPrNoGTDenom = max_func(one_val, max_sev - gt_nbSev)
    sevGTNoPrDenom = max_func(one_val, gt_nbSev)
    next_nbSevPrNoGT = (max_sev - gt_nbSev - next_nbSevPrNoGT) / sevPrNoGTDenom
    prev_nbSevPrNoGT = (max_sev - gt_nbSev - prev_nbSevPrNoGT) / sevPrNoGTDenom
    next_nbSevGTNoPr = (gt_nbSev - next_nbSevGTNoPr) / sevGTNoPrDenom
    prev_nbSevGTNoPr = (gt_nbSev - prev_nbSevGTNoPr) / sevGTNoPrDenom
    next_f1Sev = (2 * next_nbSevPrNoGT * next_nbSevGTNoPr) / (
        next_nbSevPrNoGT + next_nbSevGTNoPr + 1e-8
    )
    prev_f1Sev = (2 * prev_nbSevPrNoGT * prev_nbSevGTNoPr) / (
        prev_nbSevPrNoGT + prev_nbSevGTNoPr + 1e-8
    )
    # reverse if needed
    next_nbSevPrNoGT = -(1 - next_nbSevPrNoGT) if reverse_flag else next_nbSevPrNoGT
    prev_nbSevPrNoGT = -(1 - prev_nbSevPrNoGT) if reverse_flag else prev_nbSevPrNoGT
    next_nbSevGTNoPr = -(1 - next_nbSevGTNoPr) if reverse_flag else next_nbSevGTNoPr
    prev_nbSevGTNoPr = -(1 - prev_nbSevGTNoPr) if reverse_flag else prev_nbSevGTNoPr
    next_f1Sev = -(1 - next_f1Sev) if reverse_flag else next_f1Sev
    prev_f1Sev = -(1 - prev_f1Sev) if reverse_flag else prev_f1Sev

    # reverse patho number to take fraction
    max_pat = next_values.size(1)
    one_val = torch.ones_like(gt_nbPat)
    max_func = torch.maximum
    patPrNoGTDenom = max_func(one_val, max_pat - gt_nbPat)
    patGTNoPrDenom = max_func(one_val, gt_nbPat)
    next_nbPatPrNoGT = (max_pat - gt_nbPat - next_nbPatPrNoGT) / patPrNoGTDenom
    prev_nbPatPrNoGT = (max_pat - gt_nbPat - prev_nbPatPrNoGT) / patPrNoGTDenom
    next_nbPatGTNoPr = (gt_nbPat - next_nbPatGTNoPr) / patGTNoPrDenom
    prev_nbPatGTNoPr = (gt_nbPat - prev_nbPatGTNoPr) / patGTNoPrDenom
    next_f1Pat = (2 * next_nbPatPrNoGT * next_nbPatGTNoPr) / (
        next_nbPatPrNoGT + next_nbPatGTNoPr + 1e-8
    )
    prev_f1Pat = (2 * prev_nbPatPrNoGT * prev_nbPatGTNoPr) / (
        prev_nbPatPrNoGT + prev_nbPatGTNoPr + 1e-8
    )
    # next_nbPatPrNoGT = 1.0 - next_nbPatPrNoGT
    # prev_nbPatPrNoGT = 1.0 - prev_nbPatPrNoGT
    # reverse if needed
    next_nbPatPrNoGT = -(1 - next_nbPatPrNoGT) if reverse_flag else next_nbPatPrNoGT
    prev_nbPatPrNoGT = -(1 - prev_nbPatPrNoGT) if reverse_flag else prev_nbPatPrNoGT
    next_nbPatGTNoPr = -(1 - next_nbPatGTNoPr) if reverse_flag else next_nbPatGTNoPr
    prev_nbPatGTNoPr = -(1 - prev_nbPatGTNoPr) if reverse_flag else prev_nbPatGTNoPr
    next_f1Pat = -(1 - next_f1Pat) if reverse_flag else next_f1Pat
    prev_f1Pat = -(1 - prev_f1Pat) if reverse_flag else prev_f1Pat

    # normalize entropy
    maxH = math.log(prev_values.size(1))
    prev_ent = prev_ent / maxH
    next_ent = next_ent / maxH

    # target_entropy
    maskProbs = (
        (differential_indices != -1).float() * differential_probas if is_soft else None
    )
    target_ent = (
       1
       if not is_soft
       else torch.exp(-((maskProbs * torch.log(maskProbs + 1e-18)).sum(dim=1)))
    )

    # reverse ce if needed
    next_ce = -torch.exp(-next_ce) * target_ent if reverse_ce_flag else next_ce
    prev_ce = -torch.exp(-prev_ce) * target_ent if reverse_ce_flag else prev_ce

    # next_ce = -(-1 + next_ce) if reverse_flag and reverse_ce_flag else next_ce
    # prev_ce = -(-1 + prev_ce) if reverse_flag and reverse_ce_flag else prev_ce

    # sigmoid modulator for cross entropy
    ce_mod = sigmoid_scheduler(timestep, alpha=ce_alpha, **kwargs)

    # modulator for entropy
    ent_mod = (
        0.0
        if ent_weight == 0.0
        else (
            1.0 - ce_mod
            if ent_alpha == ce_alpha
            else 1.0 - sigmoid_scheduler(timestep, alpha=ent_alpha, **kwargs)
        )
    )

    # modulator for js
    js_mod = (
        0.0
        if js_weight == 0.0
        else (
            1.0 - ce_mod
            if js_alpha == ce_alpha
            else 1.0 - sigmoid_scheduler(timestep, alpha=js_alpha, **kwargs)
        )
    )

    # modulator for tv
    tv_mod = (
        0.0
        if tv_weight == 0.0
        else (
            1.0 - ce_mod
            if tv_alpha == ce_alpha
            else 1.0 - sigmoid_scheduler(timestep, alpha=tv_alpha, **kwargs)
        )
    )

    # modulator for patho exploration
    pat_out_mod = (
        0.0
        if pat_out_weight == 0.0
        else (
            1.0 - ce_mod
            if pat_out_alpha == ce_alpha
            else 1.0 - sigmoid_scheduler(timestep, alpha=pat_out_alpha, **kwargs)
        )
    )

    # modulator for not yet recovered patho
    pat_in_mod = (
        0.0
        if pat_in_weight == 0.0
        else (
            1.0 - ce_mod
            if pat_in_alpha == ce_alpha
            else 1.0 - sigmoid_scheduler(timestep, alpha=pat_in_alpha, **kwargs)
        )
    )

    # modulator for f1 based recovered patho
    pat_f1_mod = (
        0.0
        if pat_f1_weight == 0.0
        else (
            1.0 - ce_mod
            if pat_f1_alpha == ce_alpha
            else 1.0 - sigmoid_scheduler(timestep, alpha=pat_f1_alpha, **kwargs)
        )
    )

    # modulator for ruling out
    sev_out_mod = (
        0.0
        if sev_out_weight == 0.0
        else (
            1.0 - ce_mod
            if sev_out_alpha == ce_alpha
            else 1.0 - sigmoid_scheduler(timestep, alpha=sev_out_alpha, **kwargs)
        )
    )

    # modulator for ruling in
    sev_in_mod = (
        0.0
        if sev_in_weight == 0.0
        else (
            1.0 - ce_mod
            if sev_in_alpha == ce_alpha
            else 1.0 - sigmoid_scheduler(timestep, alpha=sev_in_alpha, **kwargs)
        )
    )

    # modulator for f1 severe pathos
    sev_f1_mod = (
        0.0
        if sev_f1_weight == 0.0
        else (
            1.0 - ce_mod
            if sev_f1_alpha == ce_alpha
            else 1.0 - sigmoid_scheduler(timestep, alpha=sev_f1_alpha, **kwargs)
        )
    )

    # modulator for severity components
    sev_mod = (
        0.0
        if (sev_ent_weight == 0.0 and sev_tv_weight == 0.0 and sev_js_weight == 0.0)
        else bell_scheduler(
            timestep, alpha_a=sev_ent_alpha, alpha_b=sev_ent_alpha_b, **kwargs
        )
    )

    # ## reward components
    r_ent = None if ent_weight == 0.0 else -((discount * next_ent) - prev_ent)
    ent = (
        0.0
        if r_ent is None
        else _clamp_utils(r_ent, bounds_dict.get("ent_min"), bounds_dict.get("ent_max"))
    )

    r_sev_out = (
        None
        if sev_out_weight == 0.0
        else (((discount * next_nbSevPrNoGT) - prev_nbSevPrNoGT) * sevPrNoGTDenom * (next_nbSevPrNoGT != prev_nbSevPrNoGT).float())
    )
    sev_out = (
        0.0
        if r_sev_out is None
        else _clamp_utils(
            r_sev_out, bounds_dict.get("sev_out_min"), bounds_dict.get("sev_out_max")
        )
    )

    r_sev_in = (
        None
        if sev_in_weight == 0.0
        else (((discount * next_nbSevGTNoPr) - prev_nbSevGTNoPr) * sevGTNoPrDenom * (next_nbSevGTNoPr != prev_nbSevGTNoPr).float())
    )
    sev_in = (
        0.0
        if r_sev_in is None
        else _clamp_utils(
            r_sev_in, bounds_dict.get("sev_in_min"), bounds_dict.get("sev_in_max")
        )
    )

    r_sev_f1 = None if sev_f1_weight == 0.0 else ((discount * next_f1Sev) - prev_f1Sev)
    sev_f1 = (
        0.0
        if r_sev_f1 is None
        else _clamp_utils(
            r_sev_f1, bounds_dict.get("sev_f1_min"), bounds_dict.get("sev_f1_max")
        )
    )

    # reward path exploration
    r_pat_out = (
        None
        if pat_out_weight == 0.0
        else ((discount * next_nbPatPrNoGT) - prev_nbPatPrNoGT) * patPrNoGTDenom
    )
    pat_out = (
        0.0
        if r_pat_out is None
        else _clamp_utils(
            r_pat_out, bounds_dict.get("pat_out_min"), bounds_dict.get("pat_out_max")
        )
    )

    r_pat_in = (
        None
        if pat_in_weight == 0.0
        else ((discount * next_nbPatGTNoPr) - prev_nbPatGTNoPr) * patGTNoPrDenom
    )
    pat_in = (
        0.0
        if r_pat_in is None
        else _clamp_utils(
            r_pat_in, bounds_dict.get("pat_in_min"), bounds_dict.get("pat_in_max")
        )
    )

    r_pat_f1 = None if pat_in_weight == 0.0 else ((discount * next_f1Pat) - prev_f1Pat)
    pat_f1 = (
        0.0
        if r_pat_f1 is None
        else _clamp_utils(
            r_pat_f1, bounds_dict.get("pat_f1_min"), bounds_dict.get("pat_f1_max")
        )
    )

    r_ce = None if ce_weight == 0.0 else -((discount * next_ce) - prev_ce)
    ce = (
        0.0
        if r_ce is None
        else _clamp_utils(r_ce, bounds_dict.get("ce_min"), bounds_dict.get("ce_max"))
    )

    # evidence based factor
    ev_factor = (1.0 - evidence.float()) if link_div_with_negative_evidence else 1.0

    # js div
    r_js = None if js_weight == 0.0 else js_div(next_p, prev_p) / MAX_JS_BOUND
    r_js = None if r_js is None else ev_factor * r_js
    js = (
        0.0
        if r_js is None
        else _clamp_utils(r_js, bounds_dict.get("js_min"), bounds_dict.get("js_max"))
    )

    # tv div
    r_tv = None if tv_weight == 0.0 else total_variation(next_p, prev_p)
    r_tv = None if r_tv is None else ev_factor * r_tv
    tv = (
        0.0
        if r_tv is None
        else _clamp_utils(r_tv, bounds_dict.get("tv_min"), bounds_dict.get("tv_max"))
    )

    # entropy for severe pathologies
    r_sev_ent = (
        None if sev_ent_weight == 0.0 else -((discount * next_sev_ent) - prev_sev_ent)
    )
    sev_ent = (
        0.0
        if r_sev_ent is None
        else _clamp_utils(
            r_sev_ent, bounds_dict.get("sev_ent_min"), bounds_dict.get("sev_ent_max")
        )
    )

    # total variation for severe pathologies
    r_sev_tv = None if sev_tv_weight == 0.0 else total_variation(next_p_sev, prev_p_sev)
    sev_tv = (
        0.0
        if r_sev_tv is None
        else _clamp_utils(
            r_sev_tv, bounds_dict.get("sev_tv_min"), bounds_dict.get("sev_tv_max")
        )
    )

    # js for severe pathologies
    r_sev_js = (
        None if sev_js_weight == 0.0 else js_div(next_p_sev, prev_p_sev) / MAX_JS_BOUND
    )
    sev_js = (
        0.0
        if r_sev_js is None
        else _clamp_utils(
            r_sev_js, bounds_dict.get("sev_js_min"), bounds_dict.get("sev_js_max")
        )
    )
    weighted_entropy = ent_weight * ent
    modulated_weighted_entropy = ent_mod * weighted_entropy
    weighted_ce = ce_weight * ce
    modulated_weighted_ce = ce_mod * weighted_ce
    weighted_js = js_weight * js
    modulated_weighted_js = js_mod * weighted_js
    weighted_tv = tv_weight * tv
    modulated_weighted_tv = tv_mod * weighted_tv
    weighted_sev_ent = sev_ent_weight * sev_ent
    modulated_weighted_sev_ent = sev_mod * weighted_sev_ent
    weighted_sev_tv = sev_tv_weight * sev_tv
    modulated_weighted_sev_tv = sev_mod * weighted_sev_tv
    weighted_sev_js = sev_js_weight * sev_js
    modulated_weighted_sev_js = sev_mod * weighted_sev_js
    weighted_sev_out = sev_out_weight * sev_out
    modulated_weighted_sev_out = sev_out_mod * weighted_sev_out
    weighted_sev_in = sev_in_weight * sev_in
    modulated_weighted_sev_in = sev_in_mod * weighted_sev_in
    weighted_sev_f1 = sev_f1_weight * sev_f1
    modulated_weighted_sev_f1 = sev_f1_mod * weighted_sev_f1
    weighted_pat_out = pat_out_weight * pat_out
    modulated_weighted_pat_out = pat_out_mod * weighted_pat_out
    weighted_pat_in = pat_in_weight * pat_in
    modulated_weighted_pat_in = pat_in_mod * weighted_pat_in
    weighted_pat_f1 = pat_f1_weight * pat_f1
    modulated_weighted_pat_f1 = pat_f1_mod * weighted_pat_f1

    log_dict = log_data(
        log_dict,
        "Entropy",
        ent_weight,
        weighted_entropy,
        modulated_weighted_entropy,
        r_ent,
        ent,
    )

    log_dict = log_data(
        log_dict,
        "CrossEntropy",
        ce_weight,
        weighted_ce,
        modulated_weighted_ce,
        r_ce,
        ce,
    )

    log_dict = log_data(
        log_dict,
        "JSDivergence",
        js_weight,
        weighted_js,
        modulated_weighted_js,
        r_js,
        js,
    )

    log_dict = log_data(
        log_dict,
        "TVDivergence",
        tv_weight,
        weighted_tv,
        modulated_weighted_tv,
        r_tv,
        tv,
    )

    log_dict = log_data(
        log_dict,
        "SeverityEntropy",
        sev_ent_weight,
        weighted_sev_ent,
        modulated_weighted_sev_ent,
        r_sev_ent,
        sev_ent,
    )

    log_dict = log_data(
        log_dict,
        "SeverityTVDivergence",
        sev_tv_weight,
        weighted_sev_tv,
        modulated_weighted_sev_tv,
        r_sev_tv,
        sev_tv,
    )

    log_dict = log_data(
        log_dict,
        "SeverityJSDivergence",
        sev_js_weight,
        weighted_sev_js,
        modulated_weighted_sev_js,
        r_sev_js,
        sev_js,
    )

    log_dict = log_data(
        log_dict,
        "SeverityRuleOutNumber",
        sev_out_weight,
        weighted_sev_out,
        modulated_weighted_sev_out,
        r_sev_out,
        sev_out,
    )

    log_dict = log_data(
        log_dict,
        "SeverityRuleInNumber",
        sev_in_weight,
        weighted_sev_in,
        modulated_weighted_sev_in,
        r_sev_in,
        sev_in,
    )

    log_dict = log_data(
        log_dict,
        "SeverityF1Number",
        sev_f1_weight,
        weighted_sev_f1,
        modulated_weighted_sev_f1,
        r_sev_f1,
        sev_f1,
    )

    log_dict = log_data(
        log_dict,
        "PathoExplorationNumber",
        pat_out_weight,
        weighted_pat_out,
        modulated_weighted_pat_out,
        r_pat_out,
        pat_out,
    )

    log_dict = log_data(
        log_dict,
        "PathoToRecoverNumber",
        pat_in_weight,
        weighted_pat_in,
        modulated_weighted_pat_in,
        r_pat_in,
        pat_in,
    )

    log_dict = log_data(
        log_dict,
        "PathoF1Number",
        pat_f1_weight,
        weighted_pat_f1,
        modulated_weighted_pat_f1,
        r_pat_f1,
        pat_f1,
    )

    value = (
        modulated_weighted_entropy
        + modulated_weighted_ce
        + modulated_weighted_js
        + modulated_weighted_tv
        + modulated_weighted_sev_ent
        + modulated_weighted_sev_tv
        + modulated_weighted_sev_js
        + modulated_weighted_sev_out
        + modulated_weighted_sev_in
        + modulated_weighted_sev_f1
        + modulated_weighted_pat_out
        + modulated_weighted_pat_in
        + modulated_weighted_pat_f1
    )
    return value, log_dict


class RewardShapingFactory:
    """A factory for evaluating different reward shaping functions.

    The predefined reward shaping functions are:
        - entropy
        - cross_entropy
        - ce_ent_sent_reshaping

    """

    def __init__(self):
        self._builders = {
            "entropy": entropy_reshaping,
            "cross_entropy": cross_entropy_reshaping,
            "ce_ent_sent_reshaping": ce_ent_sent_reshaping,
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

    def evaluate(
        self,
        key,
        next_values,
        prev_values,
        targets,
        differential_indices,
        differential_probas,
        evidence,
        discount=None,
        severity=None,
        timestep=None,
        **kwargs,
    ):
        """Evaluates a metric (defined by the provided key) given the provided data.

        Parameters
        ----------
        key: str
            key indication of which metric to use for the evaluation.
        next_values: tensor
            values characterizing the state S_{t+1}.
        prev_values: tensor
            values characterizing the state S_{t}.
        targets: tensor
            targets associated to the end goal of the trajectory (predicted pathology).
        differential_indices: tensor
            indices of the pathologies involved in the differential.
        differential_probas: tensor
            probabilities associated to the pathologies involved in the differential.
        evidence: tensor
            indicator of the simulated patient experiencing the inquired action.
        discount: float
            the discount factor. Default: None
        severity: tensor
            the severity associated to each class. It is a 1-D tensor of size num_class.
            Default: None
        timestep: tensor
            the time step (within the interaction session) associated to each element
            of the batch. Default: None

        Return
        ------
        result: tensor
            the computed auxiliary reward.
        log_data: dict
            dictionnary containing sub-component data to be logged.

        """
        builder = self._builders.get(key.lower())
        if not builder:
            raise ValueError(key + " is not a valid reward shaping metric key")
        return builder(
            next_values,
            prev_values,
            targets,
            differential_indices=differential_indices,
            differential_probas=differential_probas,
            evidence=evidence,
            discount=discount,
            severity=severity,
            timestep=timestep,
            **kwargs,
        )
