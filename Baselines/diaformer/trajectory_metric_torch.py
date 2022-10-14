import math
import torch
from torchmetrics.functional import spearman_corrcoef
import numpy as np

""" Utility functions """

def _entropy(probs, base=None, weight=None):
    if weight is None:
        entropy = - torch.sum(probs * torch.log(probs + 1e-12), dim=-1)
    else:
        entropy = - torch.sum(weight * probs * torch.log(probs + 1e-12), dim=-1)
    entropy = entropy if base is None else entropy / math.log(base)
    return entropy

def _get_pareto_frontier(xs, ys, direction="upper-left", keep_start_and_end=True):
    """Get the Pareto frontier of a set of points.

    Parameter
    ------------
    xs: iterable
        an iterable of x coordinates
    ys: iterable
        an iterable of y coordinates. Must have same length as xs
    direction: str
        which direction is the optimal direction. Default: upper-left
    keep_start_and_end: bool
        whether to always keep starting and ending points in the Pareto frontier

    Return
    ------------
    pareto_indices: numpy.array
        indices for points on the Pareto frontier
    pareto_xs: numpy.array
        sorted x coordinates of points on Pateto frontier
    pareto_ys: numpy.array
        sorted y coordinates of points on Pateto frontier
    """
    
    assert direction in {"upper-left"}
    assert len(xs) == len(ys)

    pareto_indices, pareto_xs, pareto_ys = [], [], []

    def is_on_pareto_frontier(x, y, xs, ys):
        for _x, _y in zip(xs, ys):
            if _x < x and _y > y:
                return False
        return True

    if not keep_start_and_end:
        for idx, x in enumerate(xs):
            if is_on_pareto_frontier(x, ys[idx], xs, ys):
                pareto_indices.append(idx)
                pareto_xs.append(x)
                pareto_ys.append(ys[idx])

        pareto_xs, pareto_ys = np.array(pareto_xs), np.array(pareto_ys)
        sort_indices = np.argsort(pareto_xs)
        pareto_xs, pareto_ys = pareto_xs[sort_indices], pareto_ys[sort_indices]
    else:
        for idx, x in enumerate(xs):
            if idx == 0 or idx == len(xs) - 1:
                continue
            if is_on_pareto_frontier(x, ys[idx], xs, ys):
                pareto_indices.append(idx)
                pareto_xs.append(x)
                pareto_ys.append(ys[idx])

        pareto_xs, pareto_ys = np.array(pareto_xs), np.array(pareto_ys)
        sort_indices = np.argsort(pareto_xs)
        pareto_xs, pareto_ys = pareto_xs[sort_indices], pareto_ys[sort_indices]
        pareto_indices = [0] + pareto_indices + [len(xs) - 1]
        pareto_xs = np.insert(pareto_xs, 0, xs[0])
        pareto_xs = np.insert(pareto_xs, len(pareto_xs), xs[-1])
        pareto_ys = np.insert(pareto_ys, 0, ys[0])
        pareto_ys = np.insert(pareto_ys, len(pareto_ys), ys[-1])

    return np.array(pareto_indices, dtype=np.longlong), pareto_xs, pareto_ys

""" Explore """

def new_patho_ratio(pred_probs, threshold=0.01):
    """Compute new pathologies ratio between consecutive turns from probability distributions

    The ratio of new pathologies (i.e. those that have not been included in the differentials in previous turns) in each turn. 
    
    A score of 0 is appended to the beginning of results to preserve length.

    Parameters:
    ----------
    pred_probs: torch.tensor
        a 3-d torch.tensor of predicted probabilities of pathologies. Shape: (# patients, # turns, # pathos)
    threshold: float 
        probability threshold to select differentials. Default: 0.01

    Return:
    ----------
    result: torch.tensor
        a 2-d torch.tensor of new pathology ratios at each turn starting from turn 2. A score of 0 is appended to the beginning of results. Shape: (# patients, # turns)
    
    """

    in_differentials = pred_probs > threshold

    # if the patho has been in differential in any previous turn
    in_differentials_before = torch.cumsum(
        torch.cat((torch.zeros(pred_probs.shape[0], 1, pred_probs.shape[-1], dtype=torch.bool, device=pred_probs.device), in_differentials[:, :-1, :]), dim=1), 
        dim=1).type(torch.bool)

    first_time_differential = (in_differentials & (~ in_differentials_before))

    new_differentials_ratios = first_time_differential.sum(dim=-1) / in_differentials.sum(dim=-1)
    new_differentials_ratios[:, 0] = 0  # set first turns' scores to 0

    return new_differentials_ratios

def jensenshannon_divergence(pred_probs, base=2):
    """Compute Jensen Shannon Divergence between consecutive turns from logits

    JSD is computed for each consecutive pair of distributions. A score of 0 is appended to the beginning of results to preserve length.

    Parameters:
    ----------
    pred_probs: torch.tensor
        a 3-d torch.tensor of predicted probabilities of pathologies. Shape: (# patients, # turns, # pathos)
    base: float
        the basis under which the logarith is computed. Default: 2

    Return:
    ----------
    result: torch.tensor
        a 2-d torch.tensor of JSDs at each turn starting from turn 2. A score of 0 is appended to the beginning of results. Shape: (# patients, # turns)
    
    """

    m = (pred_probs[:, :-1, :] + pred_probs[:, 1:, :]) / 2

    jsd = torch.zeros(pred_probs.shape[:-1])

    jsd[:, 1:] = 0.5 * (torch.nn.functional.kl_div(torch.log(m + 1e-12), pred_probs[:, :-1, :], reduction="none").sum(dim=-1) + \
        torch.nn.functional.kl_div(torch.log(m + 1e-12), pred_probs[:, 1:, :], reduction="none").sum(dim=-1))

    # change of base
    jsd /= math.log(base)

    return jsd


""" Confirm """

def gini(pred_probs):
    """Compute Gini coefficient for probability distributions

    Parameters:
    ----------
    pred_probs: torch.tensor
        a 3-d torch.tensor of predicted probabilities of pathologies. Shape: (# patients, # turns, # pathos)

    Return:
    ----------
    result: torch.tensor
        a 2-d torch.tensor of gini coefs at each turn. Shape: (# patients, # turns)
    
    """
    pred_probs, _ = torch.sort(pred_probs, dim=-1)
    pred_probs = (pred_probs - torch.min(pred_probs, dim=-1, keepdim=True)[0]) / (torch.max(pred_probs, dim=-1, keepdim=True)[0] - torch.min(pred_probs, dim=-1, keepdim=True)[0])
    ks = torch.arange(pred_probs.shape[-1], device=pred_probs.device) + 1
    intervals = 1 - (ks - 0.5) / pred_probs.shape[-1]
    return 1 - 2 * torch.sum(pred_probs * intervals.T, dim=-1) / pred_probs.sum(dim=-1)

def entropy_score(pred_probs):
    """Compute entropy scores for probability distributions

    Entropy score is defined as 1 - entropy(dist) / log(maximal entropy), where maximal entropy is obtained when dist is uniform, i.e. log(len(dist))
    
    Parameters:
    ----------
    pred_probs: torch.tensor
        a 3-d torch.tensor of predicted probabilities of pathologies. Shape: (# patients, # turns, # pathos)

    Return:
    ----------
    result: torch.tensor
        a 2-d torch.tensor of entropy scores at each turn. Shape: (# patients, # turns)

    """
    
    return 1 - _entropy(pred_probs, base=2) / math.log2(pred_probs.shape[-1])

""" KL score """

def kl_score(pred_probs_logits, differentials, differentials_probs=None, weight=None, alpha_schedule="linear"):
    """Calculate the KL scores.

    :math:`\alpha\exp(-KL(p_e,p))+(1-\alpha)\exp(-KL(p_c,p))`

    :math:`p_e` is an uniform distribution over all pathologies. :math:`p_c` is the differential probabilities if they are available, and an uniform distribution over differentials if they are not.

    `weight` controls the weight each pathology receives when calculating :math:`KL(p_e/p_c,p)=CE(p_e/p_c,p)-entropy(p_e/p_c)`. It can used to reflect varied severities of pathologies. 

    Parameter
    ----------
    pred_probs_logits: torch.tensor
        a 3-d torch.tensor of predicted logits of pathologies. Shape: (# patients, # turns, # pathos)
    differentials: torch.tensor
        a 2-d torch.tensor containing indices of pathologies in differentials. Shape: (# patients, # max number of differentials)
    differentials_probs: torch.tensor
        a 2-d torch.tensor containing probabilities of pathologies in differentials, in the same orders. Shape: (# patients, # max number of differentials)
    weight: torch.tensor
        a 1-d torch.tensor of pathologies' weights when calculating KL divergences. Shape: (# pathos)
    alpha_schedule: str
        alpha's decreasing schedule. Choices: {linear}, default: linear

    Return
    ----------
    result_dict: a dict of following fields
        kl_scores: the KL scores at each turn
        kl_explore: the KL divergences with the reference distribution of explore
        kl_confirm: the KL divergences with the reference distribution of confirm

    """
    assert alpha_schedule in {"linear"}
    if weight is not None:
        assert isinstance(weight, torch.Tensor)
        assert len(weight.shape) == 1 
        assert weight.shape[0] == pred_probs_logits.shape[-1]

    # build p_e
    p_e = torch.ones(pred_probs_logits.shape[-1], device=pred_probs_logits.device) / pred_probs_logits.shape[-1]

    # build p_c
    p_c = torch.zeros(pred_probs_logits.shape[0], pred_probs_logits.shape[-1] + 1, device=pred_probs_logits.device)    # add a dimension at the end for handling -1 in differentials and differentials_probs
    if differentials_probs is not None:
        p_c[torch.arange(p_c.shape[0]).unsqueeze(-1), differentials] = differentials_probs
    else:
        p_c[torch.arange(p_c.shape[0]).unsqueeze(-1), differentials] = (1 / torch.sum(differentials != -1, dim=-1)).unsqueeze(dim=1)
    p_c = p_c[:, :-1]
    assert torch.all(p_c >= 0) and torch.allclose(p_c.sum(dim=-1), torch.tensor(1, dtype=torch.float, device=p_c.device))

    # compute kls
    kl_with_p_e = torch.nn.functional.cross_entropy(pred_probs_logits.transpose(-1, -2), p_e.expand(pred_probs_logits.shape).transpose(-1, -2), weight=weight, reduction="none") \
        - _entropy(p_e, weight=weight)  # (# patients, # turns)
    kl_with_p_c = torch.nn.functional.cross_entropy(pred_probs_logits.transpose(-1, -2), p_c.unsqueeze(dim=1).expand(pred_probs_logits.shape).transpose(-1, -2), weight=weight, reduction="none") \
        - _entropy(p_c, weight=weight).unsqueeze(dim=1)  # (# patients, # turns)
    
    # get alphas
    if alpha_schedule == "linear":
        if pred_probs_logits.shape[1] > 1:
            alphas = torch.arange(start=pred_probs_logits.shape[1] - 1, end=-1, step=-1, device=pred_probs_logits.device) / (pred_probs_logits.shape[1] - 1)
        else:
            alphas = torch.tensor(0.5, device=pred_probs_logits.device)

    # compute kl scores
    kl_scores = alphas * torch.exp(- kl_with_p_e) + (1 - alphas) * torch.exp(- kl_with_p_c)

    result_dict = {
        "kl_score": kl_scores,
        "kl_explore": kl_with_p_e,
        "kl_confirm": kl_with_p_c,
    }

    return result_dict

def kl_auc(pred_probs_logits, when_ends, differentials, differentials_probs=None, weight=None):
    """Calculate the area under the curve (AUC) of the normalized KLs of explore and confirm.

    `weight` controls the weight each pathology receives when calculating :math:`KL(p_e/p_c,p)=CE(p_e/p_c,p)-entropy(p_e/p_c)`. It can used to reflect varied severities of pathologies. 

    Parameter
    ----------
    pred_probs_logits: torch.tensor
        a 3-d torch.tensor of predicted logits of pathologies. Shape: (# patients, # turns, # pathos)
    when_ends: torch.tensor
        a 1-d torch.tensor indicating at which turn each trajectory ends. Shape: (# patients)
    differentials: torch.tensor
        a 2-d torch.tensor containing indices of pathologies in differentials. Shape: (# patients, # max number of differentials)
    differentials_probs: torch.tensor
        a 2-d torch.tensor containing probabilities of pathologies in differentials, in the same orders. Shape: (# patients, # max number of differentials)
    weight: torch.tensor
        a 1-d torch.tensor of pathologies' weights when calculating KL divergences. Shape: (# pathos)

    Return
    ----------
    result_dict: a dict of following fields
        kl_auc: the AUCs of normalized KLs at each turn
        kl_explore: the KL divergences with the reference distribution of explore
        kl_confirm: the KL divergences with the reference distribution of confirm

    """
    if weight is not None:
        assert isinstance(weight, torch.Tensor)
        assert len(weight.shape) == 1 
        assert weight.shape[0] == pred_probs_logits.shape[-1]

    # build p_e
    p_e = torch.softmax(pred_probs_logits[:, 0, :].type(torch.float), dim=-1).type(pred_probs_logits.dtype) # p_e is the prediction at turn 0

    # build p_c
    p_c = torch.zeros(pred_probs_logits.shape[0], pred_probs_logits.shape[-1] + 1, device=pred_probs_logits.device, dtype=pred_probs_logits.dtype)    # add a dimension at the end for handling -1 in differentials and differentials_probs
    if differentials_probs is not None:
        p_c[torch.arange(p_c.shape[0]).unsqueeze(-1), differentials] = differentials_probs.type_as(pred_probs_logits)
    else:
        p_c[torch.arange(p_c.shape[0]).unsqueeze(-1), differentials] = (1 / torch.sum(differentials != -1, dim=-1)).unsqueeze(dim=1)
    p_c = p_c[:, :-1]
    assert torch.all(p_c >= 0) and torch.allclose(p_c.sum(dim=-1), torch.tensor(1, dtype=pred_probs_logits.dtype, device=p_c.device), atol=1e-3), f"{torch.abs(p_c.sum(dim=-1) - 1).max().item()}"

    # compute kls
    kl_with_p_e = torch.nn.functional.cross_entropy(pred_probs_logits.transpose(-1, -2).type(torch.float), p_e.unsqueeze(dim=1).expand(pred_probs_logits.shape).transpose(-1, -2).type(torch.float), weight=weight, reduction="none") \
        - _entropy(p_e.type(torch.float), weight=weight).unsqueeze(dim=1)  # (# patients, # turns)
    kl_with_p_c = torch.nn.functional.cross_entropy(pred_probs_logits.transpose(-1, -2).type(torch.float), p_c.unsqueeze(dim=1).expand(pred_probs_logits.shape).transpose(-1, -2).type(torch.float), weight=weight, reduction="none") \
        - _entropy(p_c.type(torch.float), weight=weight).unsqueeze(dim=1)  # (# patients, # turns)

    # compute aucs
    kl_aucs = torch.tensor([torch.trapezoid(1 - torch.exp(- kl_with_p_e[sample_idx, :end_idx + 1]), torch.exp(- kl_with_p_c[sample_idx, :end_idx + 1]), dim=-1) for sample_idx, end_idx in enumerate(when_ends)])

    result_dict = {
        "kl_auc": kl_aucs,
        "kl_explore": kl_with_p_e,
        "kl_confirm": kl_with_p_c,
    }

    return result_dict

def pareto_kl_auc(pred_probs_logits, when_ends, differentials, differentials_probs=None, weight=None):
    """Calculate the area under the curve (AUC) of the Pareto frontier of normalized KLs of explore and confirm.

    `weight` controls the weight each pathology receives when calculating :math:`KL(p_e/p_c,p)=CE(p_e/p_c,p)-entropy(p_e/p_c)`. It can used to reflect varied severities of pathologies. 

    Parameter
    ----------
    pred_probs_logits: torch.tensor
        a 3-d torch.tensor of predicted logits of pathologies. Shape: (# patients, # turns, # pathos)
    when_ends: torch.tensor
        a 1-d torch.tensor indicating at which turn each trajectory ends. Shape: (# patients)
    differentials: torch.tensor
        a 2-d torch.tensor containing indices of pathologies in differentials. Shape: (# patients, # max number of differentials)
    differentials_probs: torch.tensor
        a 2-d torch.tensor containing probabilities of pathologies in differentials, in the same orders. Shape: (# patients, # max number of differentials)
    weight: torch.tensor
        a 1-d torch.tensor of pathologies' weights when calculating KL divergences. Shape: (# pathos)

    Return
    ----------
    result_dict: a dict of following fields
        pareto_kl_auc: the AUCs of Pareto frontiers
        kl_explore: the KL divergences with the reference distribution of explore
        kl_confirm: the KL divergences with the reference distribution of confirm
        pareto_indices: indices of turns on the Pareto frontier

    """
    results_dict = kl_auc(pred_probs_logits, when_ends, differentials, differentials_probs=differentials_probs, weight=weight)

    xs = torch.exp(- results_dict["kl_confirm"]).cpu().numpy()
    ys = 1 - torch.exp(- results_dict["kl_explore"]).cpu().numpy()

    all_pareto_indices, pareto_kl_aucs = [], []
    for sample_idx, end_idx in enumerate(when_ends):
        pareto_indices, pareto_xs, pareto_ys = _get_pareto_frontier(xs[sample_idx, :end_idx + 1], ys[sample_idx, :end_idx + 1])
        all_pareto_indices.append(np.array(pareto_indices, dtype=np.longlong))
        pareto_kl_aucs.append(np.trapz(pareto_ys, pareto_xs))

    return_dict = {
        "pareto_kl_auc": np.array(pareto_kl_aucs),
        "kl_explore": results_dict["kl_explore"].cpu().numpy(),
        "kl_confirm": results_dict["kl_confirm"].cpu().numpy(),
        "pareto_indices": all_pareto_indices,
    }

    return return_dict

""" Summarization methods """

def summarize_spearman(scores, when_ends, ascending=True):
    """Compute the Spearman correlation between the scores and the sorted version of them.

    Parameters
    ----------
    scores: torch.tensor
        a 2-d torch.tensor of scores. Shape: (# patients, # turns)
    when_ends: torch.tensor
        a 1-d torch.tensor indicating at which turn each trajectory ends. Shape: (# patients)
    ascending: bool
        whether compare with the scores sorted in ascending or descending order. Default: ascending

    Return
    ----------
    summarized_score: torch.tensor
        a summarized score of the trajectory calculated using Spearman correlation. Shape: (# patients)

    """
    correlation_scores = [spearman_corrcoef(score[: when_ends[idx] + 1], score[: when_ends[idx] + 1].sort(dim=-1, descending=not ascending)[0]) \
        for idx, score in enumerate(scores)]

    return torch.tensor(correlation_scores, device=scores.device)

def summarize_mean(scores, when_ends, ascending=True):
    """Compute the average of scores at valid turns.

    Parameters
    ----------
    scores: torch.tensor
        a 2-d torch.tensor of scores. Shape: (# patients, # turns)
    when_ends: torch.tensor
        a 1-d torch.tensor indicating at which turn each trajectory ends. Shape: (# patients)
    ascending: bool
        for maintaining a consistent interface only, not used in calculation.

    Return
    ----------
    summarized_score: torch.tensor
        a summarized score of the trajectory. Shape: (# patients)

    """
    mask = torch.arange(scores.shape[-1], device=scores.device).repeat(scores.shape[0], 1)
    mask = mask <= when_ends.unsqueeze(-1)

    return (scores * mask).sum(dim=-1) / mask.sum(dim=-1)

""" Interface """

metrics_dict = {
    "new_patho_ratio": new_patho_ratio,
    "jsd": jensenshannon_divergence,
    "gini": gini,
    "entropy_score": entropy_score,
    "kl_score": kl_score,
    "kl_auc": kl_auc,
    "pareto_kl_auc": pareto_kl_auc,
}

metrics_whether_ascending = {
    "new_patho_ratio": False,
    "jsd": False,
    "gini": True,
    "entropy_score": True,
    "kl_explore": True,
    "kl_confirm": False,
}

summarize_dict = {
    "spearman": summarize_spearman,
    "mean": summarize_mean,
}

def evaluate_trajectory(pred_probs, when_ends=None, differentials=None, differentials_probs=None, weight=None, alpha_schedule=None, drop_first=True, which_metrics=None, summarize=None):
    """Evaluate how much trajectories of probabilities align with the desideratas. 
    
    It evaluates 2 aspects, explore and confirm, and return scores for each as a function of turn.

    Available metrics:
    - Explore:
        - new_patho_ratio: New pathology ratio
    - Confirm
        - gini: Gini coefficient
        - entropy_score: Entropy score
    - Explore and Confirm
        - kl_score: KL score

    Available summarization methods:
    - spearman: spearman correlation between the scores and the sorted version of them
    - mean: mean of all valid turns

    For KL score, if summarizing by Spearman, the summarization score is the average of the Spearman correlations of KL divergences with references for confirm and explore respectively.
    See kl_score for detail.
    
    Parameters
    ----------
    pred_probs: torch.tensor
        a 3-d torch.tensor of redicted logits of pathologies. Shape: (# patients, # turns, # pathos)
    when_ends: torch.tensor
        a 1-d torch.tensor indicating at which turn each trajectory ends. Shape: (# patients)
    differentials: torch.tensor
        a 2-d torch.tensor containing indices of pathologies in differentials. Shape: (# patients, # max number of differentials)
    differentials_probs: torch.tensor
        a 2-d torch.tensor containing probabilities of pathologies in differentials, in the same orders. Shape: (# patients, # max number of differentials)
    weight: torch.tensor
        a 1-d torch.tensor of pathologies' weights when calculating KL scores. Shape: (# pathos)
    alpha_schedule: str
        alpha's decreasing schedule. Choices: {linear}, default: linear
    drop_first: bool
        whether to drop the scores of turn 1 since "explore" scores start at turn 2. Default: True.
    which_metrics: iterable
        specifying the metrics to evaluate. By default evaluates all metrics. Default: None.
    summarize: str
        how to summarize the scores for the entire trajectory. By default they are not summarized. Default: None
    
    Return
    ----------
    output_dict: dict of metric: torch.tensor of scores/summarized score
        a dictionary of results for every metric
    """
    assert len(pred_probs.shape) == 3, "trajectory must be a 3-d torch.tensor of shape (# patients, # turns, # pathos)"
    pred_probs_logits = pred_probs  # store the raw logits for kl score
    pred_probs = torch.softmax(pred_probs.type(torch.float), dim=-1, dtype=pred_probs.dtype)
    which_metrics = set(metrics_dict.keys()) if which_metrics is None else which_metrics
    assert all(metric in metrics_dict for metric in which_metrics), f"available metrics: {set(metrics_dict.keys())}"
    assert summarize in summarize_dict or summarize is None, f"available summarization methods: {set(summarize_dict.keys())}"

    start_idx = 1 if drop_first else 0

    output_dict = {}
    for metric in which_metrics:
        if metric == "kl_score":
            assert differentials is not None, "kl score requires differentials' indices"
            alpha_schedule = "linear" if not alpha_schedule else alpha_schedule
            output = kl_score(pred_probs_logits, differentials=differentials, differentials_probs=differentials_probs, weight=weight, alpha_schedule=alpha_schedule)
            output = {key: value[:, start_idx:] for key, value in output.items()}
            output_dict.update(output)
        elif metric in {"kl_auc", "pareto_kl_auc"}:
            assert differentials is not None and when_ends is not None, f"{metric} requires differentials' indices and when_ends"
            output = metrics_dict[metric](pred_probs_logits, when_ends, differentials=differentials, differentials_probs=differentials_probs, weight=weight)
            output = {key: value[:, start_idx:] if key not in {"kl_auc", "pareto_kl_auc", "pareto_indices"} else value for key, value in output.items()}
            output_dict.update(output)
        else:
            output_dict[metric] = metrics_dict[metric](pred_probs)[:, start_idx:]

    if summarize:
        assert when_ends is not None, "when_ends is required for summarization"
        summarize_output_dict = {}
        for metric, scores in output_dict.items():
            if metric == "kl_score":
                summarize_output_dict[f"kl_explore_{summarize}"] = summarize_dict[summarize](output_dict["kl_explore"], when_ends, metrics_whether_ascending["kl_explore"])
                summarize_output_dict[f"kl_confirm_{summarize}"] = summarize_dict[summarize](output_dict["kl_confirm"], when_ends, metrics_whether_ascending["kl_confirm"])
                summarize_output_dict[f"{metric}_{summarize}"] = (summarize_output_dict[f"kl_explore_{summarize}"] + summarize_output_dict[f"kl_confirm_{summarize}"]) / 2 \
                    if summarize == "spearman" else summarize_dict[summarize](output_dict[metric], when_ends)
            elif metric in {"kl_auc", "pareto_kl_auc", "pareto_indices"}:
                summarize_output_dict[f"kl_explore_{summarize}"] = summarize_dict[summarize](output_dict["kl_explore"], when_ends, metrics_whether_ascending["kl_explore"])
                summarize_output_dict[f"kl_confirm_{summarize}"] = summarize_dict[summarize](output_dict["kl_confirm"], when_ends, metrics_whether_ascending["kl_confirm"])
                summarize_output_dict[f"{metric}_{summarize}"] = output_dict[metric]    # AUC is already summarized
            elif metric in {"kl_explore", "kl_confirm", "pareto_indices"}:
                # calculated with kl_score's summarization score
                continue
            else:
                summarize_output_dict[f"{metric}_{summarize}"] = summarize_dict[summarize](scores, when_ends, metrics_whether_ascending[metric])
        output_dict.update(summarize_output_dict)

    return output_dict