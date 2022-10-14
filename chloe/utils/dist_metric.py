import numpy as np
import sklearn


def numpy_get_severe_pathos_inout_ratio(
    predicted_differential,
    targets,
    differential_indices,
    differential_probas,
    severe_pathos,
    diff_proba_threshold=0.01,
    front_broadcast_flag=False,
    **kwargs,
):
    """Get the ratio of severe pathologies shared between GT and the predicted
    differential.
    It returns:
        - the ratio of severe pathologies that has been rule out
        - the ratio of severe pathologies that has been rule in

    It `severe_pathos` is None, it returns ration of 1 for both values.

    Parameters
    ----------
    predicted_differential: np.ndarray
        softmax values characterizing the predicted differential.
    targets: np.ndarray
        targets associated to the end goal of the trajectory (predicted pathology).
    differential_indices: np.ndarray
        indices of the pathologies involved in the differential.
    differential_probas: np.ndarray
        probabilities associated to the pathologies involved in the differential.
    severe_pathos: list, set
        collection of severe patho indices
    diff_proba_threshold: float
        the threshold for a pathology to be part of the differential. Default: 0.01
    front_broadcast_flag: bool
        flag indicating if the target should be broadcasted to the pred shape if needed
        on front dimension. Default: False

    Return
    ------
    result: tuple of 2 np.ndarray
        the ratio of severe pathologies as defined in the description.
    """
    _, nb_out, nb_in, gt_nbSev, _ = numpy_get_pathos_stats(
        predicted_differential,
        targets,
        differential_indices,
        differential_probas,
        severe_pathos,
        True,
        diff_proba_threshold,
        front_broadcast_flag,
        False,
        **kwargs,
    )
    # we dont have information on severe pathologies
    # and return 1 as ratios
    if severe_pathos is None or len(severe_pathos) == 0:
        tmp_shape = list(predicted_differential.shape)
        tmp_shape = tmp_shape[:-1]
        return (
            (np.ones(tmp_shape), np.ones(tmp_shape))
            if len(tmp_shape) > 0
            else (1.0, 1.0)
        )
    else:
        max_sev = len(severe_pathos)
        rule_out = (max_sev - gt_nbSev - nb_out) / np.maximum(1, max_sev - gt_nbSev)
        rule_in = (gt_nbSev - nb_in) / np.maximum(1, gt_nbSev)
        return rule_out, rule_in


def numpy_get_pathos_inout_ratio(
    predicted_differential,
    targets,
    differential_indices,
    differential_probas,
    diff_proba_threshold=0.01,
    front_broadcast_flag=False,
    **kwargs,
):
    """Get the ratio of pathologies shared between GT and the predicted differential.
    It returns:
        - the ratio of pathologies that has been rule out
        - the ratio of pathologies that has been rule in

    Parameters
    ----------
    predicted_differential: np.ndarray
        softmax values characterizing the predicted differential.
    targets: np.ndarray
        targets associated to the end goal of the trajectory (predicted pathology).
    differential_indices: np.ndarray
        indices of the pathologies involved in the differential.
    differential_probas: np.ndarray
        probabilities associated to the pathologies involved in the differential.
    diff_proba_threshold: float
        the threshold for a pathology to be part of the differential. Default: 0.01
    front_broadcast_flag: bool
        flag indicating if the target should be broadcasted to the pred shape if needed
        on front dimension. Default: False

    Return
    ------
    result: tuple of 2 np.ndarray
        the ratio of pathologies as defined in the description.
    """
    _, nb_out, nb_in, gt_nb, _ = numpy_get_pathos_stats(
        predicted_differential,
        targets,
        differential_indices,
        differential_probas,
        None,
        False,
        diff_proba_threshold,
        front_broadcast_flag,
        False,
        **kwargs,
    )
    max_ = predicted_differential.shape[-1]
    rule_out = (max_ - gt_nb - nb_out) / np.maximum(1, max_ - gt_nb)
    rule_in = (gt_nb - nb_in) / np.maximum(1, gt_nb)
    return rule_out, rule_in


def numpy_get_pathos_stats(
    predicted_differential,
    targets,
    differential_indices,
    differential_probas,
    severe_pathos,
    severe_flag=True,
    diff_proba_threshold=0.01,
    front_broadcast_flag=False,
    return_mask=False,
    **kwargs,
):
    """Get the stats of (severe) pathologies shared between GT and the predicted
    differential.
    It returns:
        - the stats of (severe) pathologies common to GT and Pred
        - the stats of (severe) pathologies in Pred but not in GT
        - the stats of (severe) pathologies in GT but not in Pred
        - the stats of (severe) pathologies in GT
        - the stats of (severe) pathologies in Pred

    Parameters
    ----------
    predicted_differential: np.ndarray
        softmax values characterizing the predicted differential.
    targets: np.ndarray
        targets associated to the end goal of the trajectory (predicted pathology).
    differential_indices: np.ndarray
        indices of the pathologies involved in the differential.
    differential_probas: np.ndarray
        probabilities associated to the pathologies involved in the differential.
    severe_pathos: list, set
        collection of severe patho indices
    severe_flag: boolean
        flag indicating if the computed stats is for severe pathology only.
        Default: True
    diff_proba_threshold: float
        the threshold for a pathology to be part of the differential. Default: 0.01
    front_broadcast_flag: bool
        flag indicating if the target should be broadcasted to the pred shape if needed
        on front dimension. Default: False
    return_mask: bool
        flag indicating if the value to be be returned are masks or number.
        Default: False

    Return
    ------
    result: tuple of 5 np.ndarray
        the stats of (severe) pathologies as defined in the description.
    """
    # we dont have information on severe pathologies and severe_flag is True
    # in this case, we return 0 for all stats.
    if (severe_pathos is None or len(severe_pathos) == 0) and severe_flag:
        tmp_sh1 = list(predicted_differential.shape)
        tmp_sh = tmp_sh1[:-1] if not return_mask else tmp_sh1
        tmpf = np.zeros
        return tmpf(tmp_sh), tmpf(tmp_sh), tmpf(tmp_sh), tmpf(tmp_sh), tmpf(tmp_sh)
    n = predicted_differential.shape[-1]
    if (differential_indices is not None) and (differential_probas is not None):
        tmp_shape = list(predicted_differential.shape)
        tmp_shape[-1] += 1
        gt = np.zeros(tmp_shape)
        tmp_mask = differential_indices != -1
        masked_ind = tmp_mask * differential_indices + np.logical_not(tmp_mask) * n
        if front_broadcast_flag and len(masked_ind.shape) < len(gt.shape):
            n_diff = len(gt.shape) - len(masked_ind.shape)
            tmp_shape = ([1] * n_diff) + list(masked_ind.shape)
            masked_ind = masked_ind.reshape(tmp_shape)
            differential_probas = differential_probas.reshape(tmp_shape)
        np.put_along_axis(gt, masked_ind, differential_probas, axis=-1)
        gt = gt[..., 0:-1]
    else:
        gt = np.zeros_like(predicted_differential)
        targets = np.array([targets], dtype=int)
        tgt_shape = list(targets.shape[1:]) + [1]
        targets = targets.reshape(*tgt_shape)
        if front_broadcast_flag and len(targets.shape) < len(gt.shape):
            n_diff = len(gt.shape) - len(targets.shape)
            tmp_shape = ([1] * n_diff) + list(targets.shape)
            targets = targets.reshape(*tmp_shape)
        np.put_along_axis(gt, targets, 1.0, axis=-1)

    severity_mask = np.zeros((n,), dtype=int)
    if severe_flag:
        assert severe_pathos is not None
        severity_mask[np.array(severe_pathos)] = 1  # severe pathologies
    else:
        severity_mask[:] = 1  # we consider all pathologies

    if len(gt.shape) > 1:
        tmp_shape = [1] * len(gt.shape)
        tmp_shape[-1] = -1
        severity_mask = severity_mask.reshape(*tmp_shape)

    # mask of severes patho in gt and pred
    gt_severity_mask = ((gt > diff_proba_threshold) * severity_mask).astype(bool)
    pr_severity_mask = (
        (predicted_differential > diff_proba_threshold) * severity_mask
    ).astype(bool)

    # mask of severe patho in pred and in gt
    pr_gt_common_severity_mask = (gt_severity_mask * pr_severity_mask).astype(bool)

    # mask of severe patho in pred not in gt
    pr_gt_nonshared_severity_mask = (
        np.logical_not(gt_severity_mask) * pr_severity_mask
    ).astype(bool)

    # mask of severe patho in gt not in pred
    gt_pr_nonshared_severity_mask = (
        np.logical_not(pr_severity_mask) * gt_severity_mask
    ).astype(bool)

    return (
        (
            pr_gt_common_severity_mask.astype(int).sum(-1),
            pr_gt_nonshared_severity_mask.astype(int).sum(-1),
            gt_pr_nonshared_severity_mask.astype(int).sum(-1),
            gt_severity_mask.astype(int).sum(-1),
            pr_severity_mask.astype(int).sum(-1),
        )
        if not return_mask
        else (
            pr_gt_common_severity_mask,
            pr_gt_nonshared_severity_mask,
            gt_pr_nonshared_severity_mask,
            gt_severity_mask,
            pr_severity_mask,
        )
    )


def kl_confirm_score(probas, targets, differential_indices, differential_probas, c=1):
    """
    Compute the kl_div based confirm score between predicted probas and expected target.
    This score tells how far is a predicted distribution from the target differential.

    Parameters
    ----------
    probas: np.array
        an array of size `NxC` where C is the number of classes.
        This tensor represents the proba values.
    targets: int
        the target indice. It is used only if the soft distribution is None. That is,
        `target_indices` or `target_probas` are None.
    differential_indices: np.array
        an array of size `D` where D <= C is the number of classes
        present in the soft distribution.
    differential_probas: np.array
        a tensor of same size as `target_indices` representing the probability
        associated to each class therein.
    c: float
        The weight to apply. Default: 1

    Return
    ----------
    result: list
        the computed score for each provided proba distribution.
    """
    dist = np.zeros((probas.shape[1],))
    if (differential_indices is not None) and (differential_probas is not None):
        dist[differential_indices[differential_indices != -1]] = differential_probas[
            differential_indices != -1
        ]
    else:
        dist[targets] = 1.0
    entropy = -np.sum(dist * np.log(dist + 1e-10))
    dist = dist.reshape(1, -1)
    cross_entropy = -np.sum(dist * np.log(probas + 1e-10), axis=1)
    kl_val = np.maximum(0, cross_entropy - entropy)
    kl_val = np.exp(-c * kl_val)
    return kl_val.tolist()


def kl_explore_score(probas, first_proba=None, c=1):
    """
    Compute the kl_div based explore score between predicted probas and first one.
    This score tells how far is a predicted distribution from the first prediction.

    Parameters
    ----------
    probas: np.array
        an array of size `NxC` where C is the number of classes.
        This tensor represents the proba values.
    first_proba: np.array
        the proba to compare against. If None, the first entry of `probas` will be used.
        Default: None
    c: float
        The weight to apply. Default: 1

    Return
    ----------
    result: list
        the computed score for each provided proba distribution.
    """
    dist = probas[0] if first_proba is None else first_proba
    entropy = -np.sum(dist * np.log(dist + 1e-10))
    dist = dist.reshape(1, -1)
    cross_entropy = -np.sum(dist * np.log(probas + 1e-10), axis=1)
    kl_val = np.maximum(0, cross_entropy - entropy)
    kl_val = np.exp(-c * kl_val)
    kl_val = 1.0 - kl_val
    return kl_val.tolist()


def kl_trajectory_auc(kl_explore, kl_confirm, mode="none"):
    """
    Compute the AUC of the trajectory based on kl_explore and kl_confirm.
    mode in ['sort', 'none']. If sort, we sort the kl_confirm.

    Parameters
    ----------
    kl_explore: np.array, list
        the explore scores of the predicted distribution through the trajectory.
    kl_confirm: np.array, list
        the confirm scores of the predicted distribution through the trajectory.
        Default: None
    mode: str
        The mode to be used. Default: 'none'

    Return
    ----------
    result: float, np.array
        the computed AUC.
    """
    assert mode in ["sort", "none"]
    if mode == "sort":
        tmp = np.array(list(zip(kl_explore, kl_confirm))).tolist()
        tmp = np.array(sorted(tmp, key=lambda x: x[1]))
        kl_explore, kl_confirm = tmp[:, 0], tmp[:, 1]

    kl_explore = np.array(kl_explore)
    kl_confirm = np.array(kl_confirm)
    result = np.trapz(kl_explore, kl_confirm)
    return result # np.maximum(0.0, result)


def dist_accuracy(
    pred, target, target_indices, target_probas, k, restrict_to_k=False, ignore_index=-1
):
    """Computes the fraction of the predicted topk classes in the target distribution.

    Here, we can have dirac distribution (`target`) or soft labels defined through
    the parameters `target_indices` and `target_probas`. They respectively represent
    the class indices involved in the target distribution and their corresponding
    probability. The provided `ignore_index` can be used as padding element in the
    `target_indices` field.

    Parameters
    ----------
    pred: np.array
        an array of size `C` where C is the number of classes.
        This tensor represents the logit values.
    target: int, np.array
        the target indice. It is used only if the soft distribution is None. That is,
        `target_indices` or `target_probas` are None.
    target_indices: np.array
        an array of size `D` where D <= C is the number of classes
        present in the soft distribution.
    target_probas: np.array
        a tensor of same size as `target_indices` representing the probability
        associated to each class therein.
    k: int
        The number of top element to be considered in the predicted distribution.
    restrict_to_k: bool
        Flag indicating whether or not the differential should be restricted to the topk
        values for the metric computation. Default: False
    ignore_index: int
        Specifies a target value that is ignored and does not contribute
        to the computation. Default: -1

    Return
    ----------
    result: float
        the computed fraction.

    """

    def tmp_f(a, kmin, ignore_index):
        top, ind = a[:kmin], a[kmin:]
        # mask target indices
        msk = ind != ignore_index
        # find the intersection
        rslt = np.intersect1d(top, ind)
        # compute the fraction wrt to min(kmin, len(msk))
        return 0.0 if msk.sum() == 0 else len(rslt) / min(kmin, msk.sum())

    # if not differential based, create a differential with the right number
    if (target_indices is None) or (target_probas is None):
        target_indices = np.array([target], dtype=int).reshape((-1, 1))
        target_probas = np.ones(target_indices.shape, dtype=np.float32)

    assert ignore_index < 0
    target_indices = target_indices.reshape((-1, target_indices.shape[-1]))
    target_probas = target_probas.reshape(target_indices.shape)

    if restrict_to_k:
        # sort target indices
        s_ind = np.argsort(target_probas, axis=-1)
        s_target_probas = np.take_along_axis(target_probas, s_ind[:, ::-1], axis=-1)
        s_target_indices = np.take_along_axis(target_indices, s_ind[:, ::-1], axis=-1)
        min_l = min(s_target_probas.shape[-1], k)
        target_probas = s_target_probas[:, 0:min_l]
        target_indices = s_target_indices[:, 0:min_l]

    # get the topk indices from pred distribution
    topk = np.argsort(pred, axis=-1)[..., -k:]
    kmin = min(k, pred.shape[-1])
    topk = topk.reshape((-1, kmin))

    assert (
        topk.shape[0] == target_indices.shape[0]
    ), f"{topk.shape} - {target_indices.shape} - {k} - {pred.shape}"

    merge_arr = np.concatenate((topk, target_indices), axis=1)

    # apply along axis
    result = np.apply_along_axis(tmp_f, 1, merge_arr, kmin, ignore_index)
    return np.mean(result)


def dist_ncg(
    pred, target, target_indices, target_probas, k, restrict_to_k=False, ignore_index=-1
):
    """Computes the NCG@k metric between the predicted and target distributions.

    Here, NCG means Normalized Cumulative Gain.
    We can have dirac distribution (`target`) or soft labels defined through
    the parameters `target_indices` and `target_probas`. They respectively represent
    the class indices involved in the target distribution and their corresponding
    probability. The provided `ignore_index` can be used as padding element in the
    `target_indices` field.

    Please, refer to https://en.wikipedia.org/wiki/Discounted_cumulative_gain for
    more details.

    Parameters
    ----------
    pred: np.array
        an array of size `C` where C is the number of classes.
        This tensor represents the logit values.
    target: int, np.array
        the target indice. It is used only if the soft distribution is None. That is,
        `target_indices` or `target_probas` are None.
    target_indices: np.array
        an array of size `D` where D <= C is the number of classes
        present in the soft distribution.
    target_probas: np.array
        a tensor of same size as `target_indices` representing the probability
        associated to each class therein.
    k: int
        The number of top element to be considered in the predicted distribution.
    restrict_to_k: bool
        Flag indicating whether or not the differential should be restricted to the topk
        values for the metric computation. Default: False
    ignore_index: int
        Specifies a target value that is ignored and does not contribute
        to the computation. Default: -1

    Return
    ----------
    result: float
        the computed fraction.

    """

    def tmp_f(a, kmin, kmin2, ignore_index):
        top, ind, rel = a[:kmin], a[kmin : kmin + kmin2], a[kmin + kmin2 :]
        top = top.astype(int)
        ind = ind.astype(int)
        # mask target indices
        msk = ind != ignore_index
        # dict
        tmp_dict = {ind[i]: rel[i] for i in range(len(ind)) if ind[i] != ignore_index}
        # find the intersection
        rslt = np.intersect1d(top, ind)
        # get the min_len
        min_len = min(kmin, msk.sum())
        # compute the fraction components
        denominator = rel[:min_len].sum()
        numerator = sum([tmp_dict[i] for i in rslt])
        # compute the fraction wrt to min(kmin, len(msk))
        return 0.0 if min_len == 0 else numerator / denominator

    # if not differential based, create a differential with the right number
    if (target_indices is None) or (target_probas is None):
        target_indices = np.array([target], dtype=int).reshape((-1, 1))
        target_probas = np.ones(target_indices.shape, dtype=np.float32)

    # mask target indices
    assert ignore_index < 0
    target_indices = target_indices.reshape((-1, target_indices.shape[-1]))
    target_probas = target_probas.reshape((-1, target_probas.shape[-1]))
    mask = target_indices != ignore_index

    # relevance
    relevance = 2.0 ** (target_probas) - 1.0
    relevance[~mask] = 0.0

    # sort target indices
    s_ind = np.argsort(relevance, axis=-1)
    sorted_target_relevance = np.take_along_axis(relevance, s_ind[:, ::-1], axis=-1)
    sorted_target_indices = np.take_along_axis(target_indices, s_ind[:, ::-1], axis=-1)

    if restrict_to_k:
        min_l = min(sorted_target_indices.shape[-1], k)
        sorted_target_relevance = sorted_target_relevance[:, 0:min_l]
        sorted_target_indices = sorted_target_indices[:, 0:min_l]

    # get the topk indices from pred distribution
    topk = np.argsort(pred, axis=-1)[..., -k:]
    kmin = min(k, pred.shape[-1])
    topk = topk.reshape((-1, kmin))
    kmin2 = sorted_target_indices.shape[-1]

    assert topk.shape[0] == sorted_target_indices.shape[0]
    assert topk.shape[0] == sorted_target_relevance.shape[0]

    merge_arr = np.concatenate(
        (topk, sorted_target_indices, sorted_target_relevance), axis=-1
    )

    # apply along axis
    result = np.apply_along_axis(tmp_f, 1, merge_arr, kmin, kmin2, ignore_index)
    return np.mean(result)


def dist_ndcg(
    pred, target, target_indices, target_probas, k, restrict_to_k=False, ignore_index=-1
):
    """Computes the NDCG@k metric between the predicted and target distributions.

    Here, NDCG means Normalized Discounted Cumulative Gain.
    We can have dirac distribution (`target`) or soft labels defined through
    the parameters `target_indices` and `target_probas`. They respectively represent
    the class indices involved in the target distribution and their corresponding
    probability. The provided `ignore_index` can be used as padding element in the
    `target_indices` field.

    Please, refer to https://en.wikipedia.org/wiki/Discounted_cumulative_gain for
    more details.

    Parameters
    ----------
    pred: np.array
        an array of size `C` where C is the number of classes.
        This tensor represents the logit values.
    target: int, np.array
        the target indice. It is used only if the soft distribution is None. That is,
        `target_indices` or `target_probas` are None.
    target_indices: np.array
        an array of size `D` where D <= C is the number of classes
        present in the soft distribution.
    target_probas: np.array
        a tensor of same size as `target_indices` representing the probability
        associated to each class therein.
    k: int
        The number of top element to be considered in the predicted distribution.
    restrict_to_k: bool
        Flag indicating whether or not the differential should be restricted to the topk
        values for the metric computation. Default: False
    ignore_index: int
        Specifies a target value that is ignored and does not contribute
        to the computation. Default: -1

    Return
    ----------
    result: float
        the computed metric.

    """

    def tmp_f(a, kmin, msk_len):
        topk_target, topk_pred = a[:kmin], a[kmin:]
        # mask target indices
        # find the diff
        rslt = np.setdiff1d(topk_pred, topk_target)
        mask = np.zeros(msk_len, dtype=topk_pred.dtype)
        mask[rslt] = 1
        # compute the fraction wrt to min(kmin, len(msk))
        return mask

    pred = pred.reshape((-1, pred.shape[-1]))
    distrib = np.zeros_like(pred, dtype=np.float32)
    # if not differential based, create a differential with the right number
    if (target_indices is None) or (target_probas is None):
        target = np.array([target], dtype=int)
        target = target.reshape(-1, 1)
        np.put_along_axis(distrib, target, 1.0, axis=-1)
    else:
        target_indices = target_indices.reshape((-1, target_indices.shape[-1]))
        target_probas = target_probas.reshape((-1, target_probas.shape[-1]))
        # mask target indices
        mask = target_indices != ignore_index
        for i in range(distrib.shape[0]):
            ind = target_indices[i][mask[i]]
            val = target_probas[i][mask[i]]
            distrib[i, ind] = val

    # normalize
    distrib = distrib / distrib.sum(axis=-1, keepdims=True)

    # use the following gain function: 2**(rel) -1
    relevance = 2.0 ** (distrib) - 1.0

    if restrict_to_k:
        # sort target indices
        topk_target = np.argsort(relevance, axis=-1)[..., -k:]
        topk_pred = np.argsort(pred, axis=-1)[..., -k:]
        # merge array
        merge_arr = np.concatenate((topk_target, topk_pred), axis=-1)
        # get the mask
        kmin = topk_target.shape[1]
        mask_int = np.apply_along_axis(tmp_f, 1, merge_arr, kmin, relevance.shape[-1])
        mask = mask_int != 0
        # modify the relevancy value
        np.putmask(relevance, mask, 0)

    isfinite = np.isfinite(pred).all()
    return sklearn.metrics.ndcg_score(relevance, pred, k=k) if isfinite else 0


def numpy_softmax(x, axis=-1):
    """Compute softmax values for each sets of scores in x.

    Parameters
    ----------
    x: np.array
        an array representing the logit values.
    axis: int
        the axis to normalize accross. Default: -1

    Return
    ----------
    result: np.array
        input with softmax applied on the specified axis.

    """
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


def numpy_logsoftmax(x, axis=-1):
    """Compute softmax values for each sets of scores in x.

    Parameters
    ----------
    x: np.array
        an array representing the logit values.
    axis: int
        the axis to normalize accross. Default: -1

    Return
    ----------
    result: np.array
        input with log softmax applied on the specified axis.

    """
    v = x - np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(v)
    return v - np.log(e_x.sum(axis=axis, keepdims=True))


def dist_kl_div(pred1, pred2, axis=-1):
    """Computes the KL divergence between two predicted distribution.

    Here, distribution are provided in form of logits values.
    the computed metric is KL(pred1 || pred2).

    Parameters
    ----------
    pred1: np.array
        an array of size `C` where C is the number of classes.
        This tensor represents the logit values.
    pred2: np.array
        an array of size `C` where C is the number of classes.
        This tensor represents the logit values.
    axis: int
        the axis to normalize accross. Default: -1

    Return
    ----------
    result: float
        the computed kl divergence.

    """
    log_p1 = numpy_logsoftmax(pred1, axis=axis)
    p1 = np.exp(log_p1)
    log_p2 = numpy_logsoftmax(pred2, axis=axis)
    assert log_p1.shape == log_p2.shape
    log_p1[log_p1 == -np.inf] = 0.0
    log_p1[log_p2 == -np.inf] = 0.0
    log_p2[log_p2 == -np.inf] = 0.0

    result = p1 * (log_p1 - log_p2)

    return result.sum(axis=axis)


def dist_js_div(pred1, pred2, axis=-1, base=2):
    """Computes the JS divergence between two predicted distribution.

    Here, distribution are provided in form of logits values.
    the computed metric is JS(pred1 || pred2).

    Parameters
    ----------
    pred1: np.array
        an array of size `C` where C is the number of classes.
        This tensor represents the logit values.
    pred2: np.array
        an array of size `C` where C is the number of classes.
        This tensor represents the logit values.
    axis: int
        the axis to normalize accross. Default: -1
    base: float
        the basis under which the logarith is computed. Default: 2

    Return
    ----------
    result: float
        the computed js divergence.

    """
    log_p1 = numpy_logsoftmax(pred1, axis=axis)
    log_p2 = numpy_logsoftmax(pred2, axis=axis)
    p1 = np.exp(log_p1)
    p2 = np.exp(log_p2)
    m = (p1 + p2) / 2
    log_m = np.log(m + 1e-12)
    log_p1[log_p1 == -np.inf] = 0.0
    log_p2[log_p2 == -np.inf] = 0.0

    r1 = 0.5 * (p1 * (log_p1 - log_m)).sum(axis=axis)
    r2 = 0.5 * (p2 * (log_p2 - log_m)).sum(axis=axis)

    result = (r1 + r2) / np.log(base)
    max_value = np.log(2) / np.log(base)
    if isinstance(result, np.ndarray):
        result[result > max_value] = max_value
    else:
        if result > max_value:
            result = max_value
    return result


def dist_total_variation(pred1, pred2, axis=-1):
    """Computes the total variation between two predicted distribution.

    Here, distribution are provided in form of logits values.

    Parameters
    ----------
    pred1: np.array
        an array of size `C` where C is the number of classes.
        This tensor represents the logit values.
    pred2: np.array
        an array of size `C` where C is the number of classes.
        This tensor represents the logit values.
    axis: int
        the axis to normalize accross. Default: -1

    Return
    ----------
    result: float
        the computed total variation.

    """
    p1 = numpy_softmax(pred1, axis=axis)
    p2 = numpy_softmax(pred2, axis=axis)

    return np.abs(p1 - p2).sum(axis=axis) / 2.0
