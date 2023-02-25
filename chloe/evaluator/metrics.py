import json
import numpy as np


def write_json(data, fp):
    with open(fp, "w") as outfile:
        json.dump(data, outfile, indent=4)

def kl_div(log_m, tgt_p, axis=-1):
    log_tgt_p = np.log(tgt_p + 1e-12)
    return (tgt_p * (log_tgt_p - log_m)).sum(axis=axis)

def js_div(src_p, tgt_p, axis=-1):
    m = (src_p + tgt_p) / 2.0
    log_m = np.log(m + 1e-12)
    result = 0.5 * kl_div(log_m, src_p, axis) + 0.5 * kl_div(log_m, tgt_p, axis)
    max_value = np.log(2)
    if isinstance(result, np.ndarray):
        result[result > max_value] = max_value
    else:
        if result > max_value:
            result = max_value
    return result / max_value

def get_average_state_from_percent(data, percent=0.0, end_percent=1.0, normalize=True):
    """Get the average stat from a data sequence starting at a given percent.
    Parameters
    ----------
    data:  sequence
        the provided data.
    percent: float, list, tuple
        the provided percentage. Default: 0.0.
    end_percent: float, list, tuple
        the provided end percentage. Default: 1.0.
    normalize: boolean
        whether to normalize
    Returns
    -------
    result: float, list
        the computed average stat.
    """
    assert not (percent is None and end_percent is None)
    if percent is None:
        percent = end_percent
    elif end_percent is None:
        end_percent = percent
    assert isinstance(percent, (int, float, list, tuple))
    assert isinstance(end_percent, (int, float, list, tuple))
    extract_flag = False
    if isinstance(percent, (int, float)) and isinstance(end_percent, (int, float)):
        percent = [percent]
        end_percent = [end_percent]
        extract_flag = True
    elif isinstance(percent, (int, float)):
        percent = [percent] * len(end_percent)
    elif isinstance(end_percent, (int, float)):
        end_percent = [end_percent] * len(percent)
    assert len(percent) == len(end_percent)
    length = len(data)
    result = []
    for b, e in zip(percent, end_percent):
        assert b >= 0.0 and b <= 1.0
        assert e >= 0.0 and e <= 1.0
        assert e >= b
        i = int(length * b)
        j = int(length * e)
        j = j + 1 if j == i else j
        j = min(length, j)
        i = i - 1 if i == length else i
        n = j - i
        r = sum(data[i:j])
        r = r / max(1, n) if normalize else r
        result.append(r)
    if len(result) == 1 and extract_flag:
        result = result[0]
    return result

def compute_severity_stats(preds, gt_diff, severity_mask, diff_proba_threshold):
    if severity_mask is None:
        severity_mask = [0] * preds.shape[-1]
    if len(preds.shape) > len(gt_diff.shape):
        tmp_shape = ([1] * (len(preds.shape) - len(gt_diff.shape))) + list(gt_diff.shape)
        gt_diff = gt_diff.reshape(tmp_shape)
    sev_shape = ([1] * (len(gt_diff.shape) - 1)) + [-1]
    severity_mask= np.array(severity_mask).reshape(sev_shape)
    gt_severity_mask = ((gt_diff > diff_proba_threshold) * severity_mask).astype(bool)
    pr_severity_mask = ((preds > diff_proba_threshold) * severity_mask).astype(bool)
    # mask of severe patho in pred not in gt
    pr_gt_nonshared_severity_mask = (np.logical_not(gt_severity_mask) * pr_severity_mask).astype(bool)
    # mask of severe patho in gt not in pred
    gt_pr_nonshared_severity_mask = (np.logical_not(pr_severity_mask) * gt_severity_mask).astype(bool)
    nb_out = pr_gt_nonshared_severity_mask.sum(-1)
    nb_in = gt_pr_nonshared_severity_mask.sum(-1)
    gt_nbSev = gt_severity_mask.astype(int).sum(-1)
    max_sev = severity_mask.sum()
    pred_no_gt = (max_sev - gt_nbSev - nb_out) / np.maximum(1, max_sev - gt_nbSev)
    gt_no_pred = (gt_nbSev - nb_in) / np.maximum(1, gt_nbSev)
    gt_pred_f1 = compute_f1(pred_no_gt, gt_no_pred)
    return pred_no_gt, gt_no_pred, gt_pred_f1
    

def compute_exploration_reward(probas, weightExploration, explorationTemporalWeight, min_exploration_reward, max_exploration_reward):
    if probas.shape[1] <= 1:
        return 0.0
    prev_p = probas[:, 0:-1]
    next_p = probas[:, 1:]
    reward = js_div(next_p, prev_p)
    if min_exploration_reward is not None:
        reward[reward < min_exploration_reward] = min_exploration_reward
    if max_exploration_reward is not None:
        reward[reward > max_exploration_reward] = max_exploration_reward
    return  weightExploration * reward * np.array(explorationTemporalWeight[:reward.shape[1]]).reshape((1,-1))
    
def compute_confirmation_reward(probas, discount, target_dist, weightConfirmation, confirmationTemporalWeight, min_confirmation_reward, max_confirmation_reward):
    if probas.shape[1] <= 1:
        return 0.0
    if len(probas.shape) > len(target_dist.shape):
        tmp_shape = list(target_dist.shape[0:-1])  + ([1] * (len(probas.shape) - len(target_dist.shape))) + [target_dist.shape[-1]]
        target_dist = target_dist.reshape(tmp_shape)
    cross_entropy = -np.sum(target_dist * np.log(probas + 1e-10), axis=-1)
    prev_p = cross_entropy[:, 0:-1]
    next_p = cross_entropy[:, 1:]
    reward = -((discount * next_p) - prev_p)
    if min_confirmation_reward is not None:
        reward[reward < min_confirmation_reward] = min_confirmation_reward
    if max_confirmation_reward is not None:
        reward[reward > max_confirmation_reward] = max_confirmation_reward
    return  weightConfirmation * reward * np.array(confirmationTemporalWeight[:reward.shape[1]]).reshape((1,-1))
    
    
def compute_severity_reward(probas, discount, target_dist, weightSeverity, min_severity_reward, max_severity_reward, severity_mask, treshold):
    if probas.shape[1] <= 1:
        return 0.0
    if len(probas.shape) > len(target_dist.shape):
        tmp_shape = list(target_dist.shape[0:-1])  + ([1] * (len(probas.shape) - len(target_dist.shape))) + [target_dist.shape[-1]]
        target_dist = target_dist.reshape(tmp_shape)
    if len(probas.shape) > len(severity_mask.shape):
        tmp_shape = ([1] * (len(probas.shape) - len(severity_mask.shape))) + list(severity_mask.shape)
        severity_mask = severity_mask.reshape(tmp_shape)
    
    gt_mask = (target_dist > treshold).astype(bool)
    proba_mask = (probas > treshold).astype(bool)
    
    sev_gt_mask = (gt_mask * severity_mask).astype(bool)
    sev_proba_mask = (proba_mask * severity_mask).astype(bool)
    
    sev_not_gt_not_proba = (np.logical_not(sev_gt_mask) * np.logical_not(sev_proba_mask) * severity_mask).astype(int)
    
    sevOut = sev_not_gt_not_proba.sum(axis=-1)
    prev_p = sevOut[:, 0:-1]
    next_p = sevOut[:, 1:]
    reward = ((discount * next_p) - prev_p) * (next_p != prev_p)
    if min_severity_reward is not None:
        reward[reward < min_severity_reward] = min_severity_reward
    if max_severity_reward is not None:
        reward[reward > max_severity_reward] = max_severity_reward
    return  weightSeverity * reward
    
def compute_classification_reward(probas, target_dist, weightClassification, min_classification_reward, max_classification_reward, weightSevIn, severity_mask, treshold):
    if len(probas.shape) > len(target_dist.shape):
        tmp_shape = list(target_dist.shape[0:-1])  + ([1] * (len(probas.shape) - len(target_dist.shape))) + [target_dist.shape[-1]]
        target_dist = target_dist.reshape(tmp_shape)
    if len(probas.shape) > len(severity_mask.shape):
        tmp_shape = ([1] * (len(probas.shape) - len(severity_mask.shape))) + list(severity_mask.shape)
        severity_mask = severity_mask.reshape(tmp_shape)
    
    gt_mask = (target_dist > treshold).astype(bool)
    proba_mask = (probas > treshold).astype(bool)
    
    sev_gt_mask = (gt_mask * severity_mask).astype(bool)
    sev_proba_mask = (proba_mask * severity_mask).astype(bool)
    
    sevIn = (sev_gt_mask * sev_proba_mask).astype(int).sum(axis=-1)
    sevY = sev_gt_mask.astype(int).sum(axis=-1)
    
    cross_entropy = -np.sum(target_dist * np.log(probas + 1e-10), axis=-1)
    reward = -(cross_entropy) + (weightSevIn * (sevIn/np.maximum(1, sevY)))
    if min_classification_reward is not None:
        reward[reward < min_classification_reward] = min_classification_reward
    if max_classification_reward is not None:
        reward[reward > max_classification_reward] = max_classification_reward
    return  weightClassification * reward
    

def kl_trajectory_auc(kl_explore, kl_confirm):
    kl_explore = np.array(kl_explore)
    kl_confirm = np.array(kl_confirm)
    result = np.trapz(kl_explore, kl_confirm, axis=-1)
    return result

def kl_confirm_score(probas, dist, c=1):
    entropy = -np.sum(dist * np.log(dist + 1e-10), axis=-1)
    if len(probas.shape) > len(dist.shape):
        tmp_shape = ([1] * (len(probas.shape) - len(dist.shape))) + list(dist.shape)
        dist = dist.reshape(tmp_shape)
    cross_entropy = -np.sum(dist * np.log(probas + 1e-10), axis=-1)
    kl_val = np.maximum(0, cross_entropy - entropy)
    kl_val = np.exp(-c * kl_val)
    return kl_val


def kl_explore_score(probas, first_proba=None, c=1):
    dist = probas[0] if first_proba is None else first_proba
    entropy = -np.sum(dist * np.log(dist + 1e-10), axis=-1)
    if len(probas.shape) > len(dist.shape):
        tmp_shape = ([1] * (len(probas.shape) - len(dist.shape))) + list(dist.shape)
        dist = dist.reshape(tmp_shape)
    cross_entropy = -np.sum(dist * np.log(probas + 1e-10), axis=-1)
    kl_val = np.maximum(0, cross_entropy - entropy)
    kl_val = np.exp(-c * kl_val)
    kl_val = 1.0 - kl_val
    return kl_val
    
def kl_trajectory_score(kl_explore, kl_confirm, alphas=None):
    kl_explore = np.array(kl_explore)
    kl_confirm = np.array(kl_confirm)
    if alphas is None:
        if kl_explore.shape[-1] > 1:
            alphas = np.arange(start=kl_explore.shape[-1] - 1, stop=-1, step=-1)
            alphas = alphas / (kl_explore.shape[-1] - 1)
            if len(kl_explore.shape) > len(alphas.shape):
                tmp_shape = ([1] * (len(kl_explore.shape) - len(alphas.shape))) + list(alphas.shape)
                alphas = alphas.reshape(tmp_shape)
        else:
            alphas = 0.5
    score = alphas * kl_explore + (1 - alphas) * kl_confirm
    return score

def compute_f1(p, r):
    if isinstance(p, (list, tuple)):
        p = np.array(p)
    if isinstance(r, (list, tuple)):
        r = np.array(r)
    denom = p + r
    return (2 * p * r) / (denom + 1e-10)
    

def compute_metrics(
    gt_differential, disease, final_diags, all_diags, batch_env_reward, valid_timesteps,
    present_evidences, inquired_evidences, symptom_mask, atcd_mask, severity_mask, tres=0.01,
    **kwargs
):
        explorationTemporalWeight = kwargs.get("explorationTemporalWeight")
        weightExploration = kwargs.get("weightExploration")
        min_exploration_reward = kwargs.get("min_exploration_reward")
        max_exploration_reward = kwargs.get("max_exploration_reward")
        
        confirmationTemporalWeight = kwargs.get("confirmationTemporalWeight")
        weightConfirmation = kwargs.get("weightConfirmation")
        min_confirmation_reward = kwargs.get("min_confirmation_reward")
        max_confirmation_reward = kwargs.get("max_confirmation_reward")

        weightSeverity = kwargs.get("weightSeverity")
        min_severity_reward = kwargs.get("min_severity_reward")
        max_severity_reward = kwargs.get("max_severity_reward")

        weightClassification = kwargs.get("weightClassification")
        min_classification_reward = kwargs.get("min_classification_reward")
        max_classification_reward = kwargs.get("max_classification_reward")
        weightSevIn = kwargs.get("weightSevIn")
        
        discount = kwargs.get("discount")
        discountFactor = None if discount is None else np.cumprod([1] + ([discount] * (valid_timesteps.shape[1]))).reshape((1, -1))
        
        
        all_indices = list(range(disease.shape[0]))
        top_ranked = np.argsort(final_diags, axis=-1)
        top_ranked = top_ranked[:, ::-1]
        gt_diff_top_ranked = np.argsort(gt_differential, axis=-1)
        gt_diff_top_ranked = gt_diff_top_ranked[:, ::-1]
        all_len = np.sum(valid_timesteps, axis=-1) + 1

        result = {}
        result["IL"] = np.mean(all_len)
        result["GTPA"] = np.mean(final_diags[all_indices, disease] > tres)
        result["GTPA@1"] = np.mean(np.logical_and(disease == top_ranked[:, 0], final_diags[all_indices, disease] > tres))
        result["GTPA@3"] = np.mean(np.logical_and(np.any(disease.reshape(-1, 1) == top_ranked[:, 0:3], axis=-1), final_diags[all_indices, disease] > tres))
        result["GTPA@5"] = np.mean(np.logical_and(np.any(disease.reshape(-1, 1) == top_ranked[:, 0:5], axis=-1), final_diags[all_indices, disease] > tres))

        totalReward = (batch_env_reward * valid_timesteps)
        result["EnvReward"] = np.mean((batch_env_reward * valid_timesteps).sum(axis=1))
        if discount is not None:
            result["DiscountedEnvReward"] = np.mean((batch_env_reward * valid_timesteps * discountFactor[:, 0:batch_env_reward.shape[1]]).sum(axis=1))
        if (explorationTemporalWeight is not None) and (weightExploration is not None):
            explReward = compute_exploration_reward(all_diags, weightExploration, explorationTemporalWeight, min_exploration_reward, max_exploration_reward)
            totalReward += (explReward * valid_timesteps)
            result["ExplorationReward"] = np.mean((explReward * valid_timesteps).sum(axis=1))
            if discount is not None:
                result["DiscountedExplorationReward"] = np.mean((explReward * valid_timesteps * discountFactor[:, 0:explReward.shape[1]]).sum(axis=1))
        if (confirmationTemporalWeight is not None) and (discount is not None) and (weightConfirmation is not None):
            confReward = compute_confirmation_reward(
                all_diags, discount, gt_differential, weightConfirmation, confirmationTemporalWeight, min_confirmation_reward, max_confirmation_reward
            )
            totalReward += (confReward * valid_timesteps)
            result["ConfirmationReward"] = np.mean((confReward * valid_timesteps).sum(axis=1))
            if discount is not None:
                result["DiscountedConfirmationReward"] = np.mean((confReward * valid_timesteps * discountFactor[:, 0:confReward.shape[1]]).sum(axis=1))
        if (discount is not None) and (weightSeverity is not None):
            sevReward = compute_severity_reward(all_diags, discount, gt_differential, weightSeverity, min_severity_reward, max_severity_reward, severity_mask, tres)
            totalReward += (sevReward * valid_timesteps)
            result["SeverityReward"] = np.mean((sevReward * valid_timesteps).sum(axis=1))
            if discount is not None:
                result["DiscountedSeverityReward"] = np.mean((sevReward * valid_timesteps * discountFactor[:, 0:sevReward.shape[1]]).sum(axis=1))
        if (weightSevIn is not None) and (weightClassification is not None):
            clfReward = compute_classification_reward(
                all_diags, gt_differential, weightClassification, min_classification_reward, max_classification_reward, weightSevIn, severity_mask, tres
            )
            tmpValid = np.zeros((all_diags.shape[0], all_diags.shape[1]))
            tmpValid[np.arange(all_diags.shape[0]), valid_timesteps.sum(axis=1)] = 1
            totalReward = np.concatenate((totalReward, np.zeros((all_diags.shape[0], 1))), axis=1)
            totalReward += (clfReward * tmpValid)
            result["ClassificationReward"] = np.mean((clfReward * tmpValid).sum(axis=1))
            
        result["TotalReward"] = np.mean((totalReward).sum(axis=1))
        if discount is not None:
            result["DiscountedTotalReward"] = np.mean((totalReward * discountFactor[:, 0:totalReward.shape[1]]).sum(axis=1))

        gt_diff_mask = (gt_differential > tres)
        pred_diff_mask = (final_diags > tres)

        ddr = np.sum(np.logical_and(gt_diff_mask, pred_diff_mask), axis=-1) / np.maximum(1, np.sum(gt_diff_mask, axis=-1))
        ddp = np.sum(np.logical_and(gt_diff_mask, pred_diff_mask), axis=-1) / np.maximum(1, np.sum(pred_diff_mask, axis=-1))
        ddf1 = compute_f1(ddp, ddr)

        result["DDR"] = np.mean(ddr)
        result["DDP"] = np.mean(ddp)
        result["DDF1"] = np.mean(ddf1)
        for k in [1, 3, 5]:
            tmp_gt_k = np.zeros_like(gt_diff_mask).astype(bool)
            np.put_along_axis(tmp_gt_k, gt_diff_top_ranked[:, 0:k], True, 1)
            tmp_gt_k = np.logical_and(gt_diff_mask, tmp_gt_k)

            tmp_pred_k = np.zeros_like(pred_diff_mask).astype(bool)
            np.put_along_axis(tmp_pred_k, top_ranked[:, 0:k], True, 1)
            tmp_pred_k = np.logical_and(pred_diff_mask, tmp_pred_k)

            tmp_ddr = np.sum(np.logical_and(tmp_gt_k, tmp_pred_k), axis=-1) / np.maximum(1, np.sum(tmp_gt_k, axis=-1))
            tmp_ddp = np.sum(np.logical_and(tmp_gt_k, tmp_pred_k), axis=-1) / np.maximum(1, np.sum(tmp_pred_k, axis=-1))
            tmp_ddf1 = compute_f1(tmp_ddp, tmp_ddr)
            result[f"DDR@{k}"] = np.mean(tmp_ddr)
            result[f"DDP@{k}"] = np.mean(tmp_ddp)
            result[f"DDF1@{k}"] = np.mean(tmp_ddf1)

        dsp, dsr, dsf1 = compute_severity_stats(final_diags, gt_differential, severity_mask, tres)
        result["DSP"] = np.mean(dsp)
        result["DSR"] = np.mean(dsr)
        result["DSF1"] = np.mean(dsf1)

        pos_evi = np.logical_and(present_evidences, inquired_evidences)
        per = np.sum(pos_evi, axis=-1) / np.maximum(1, np.sum(present_evidences, axis=-1))
        pep = np.sum(pos_evi, axis=-1) / np.maximum(1, np.sum(inquired_evidences, axis=-1))
        pef1 = compute_f1(pep, per)

        result["PER"] = np.mean(per)
        result["PEP"] = np.mean(pep)
        result["PEF1"] = np.mean(pef1)

        present_symptoms = np.logical_and(present_evidences, symptom_mask.reshape(1, -1))
        inquired_symptoms = np.logical_and(inquired_evidences, symptom_mask.reshape(1, -1))
        pos_symp = np.logical_and(present_symptoms, inquired_symptoms)
        psr = np.sum(pos_symp, axis=-1) / np.maximum(1, np.sum(present_symptoms, axis=-1))
        psp = np.sum(pos_symp, axis=-1) / np.maximum(1, np.sum(inquired_symptoms, axis=-1))
        psf1 = compute_f1(psp, psr)

        result["PSR"] = np.mean(psr)
        result["PSP"] = np.mean(psp)
        result["PSF1"] = np.mean(psf1)

        present_atcds = np.logical_and(present_evidences, atcd_mask.reshape(1, -1))
        inquired_atcds = np.logical_and(inquired_evidences, atcd_mask.reshape(1, -1))
        pos_atcd = np.logical_and(present_atcds, inquired_atcds)
        par = np.sum(pos_atcd, axis=-1) / np.maximum(1, np.sum(present_atcds, axis=-1))
        pap = np.sum(pos_atcd, axis=-1) / np.maximum(1, np.sum(inquired_atcds, axis=-1))
        paf1 = compute_f1(pap, par)

        result["PAR"] = np.mean(par)
        result["PAP"] = np.mean(pap)
        result["PAF1"] = np.mean(paf1)


        tmp_shape = list(gt_differential.shape)
        tmp_shape = tmp_shape[0:1] + [1] + tmp_shape[1:]
        gt_diff_proba = gt_differential.reshape(tmp_shape)
        confirm_score = kl_confirm_score(all_diags, gt_diff_proba)
        explore_score = kl_explore_score(all_diags, first_proba=all_diags[:, 0:1])
        succesive_explore_score = kl_explore_score(all_diags[:, 1:], first_proba=all_diags[:, 0:-1])

        pred_no_gt, gt_no_pred, gt_pred_f1 = compute_severity_stats(all_diags, gt_diff_proba, severity_mask, tres)

        p = list(range(0, 105, 5))
        # p_idx = {v: i for i, v in enumerate(p)}
        p = [v / 100.0 for v in p]
        
        t_explore_score = 0
        t_succesive_explore_score = 0
        t_confirm_score = 0
        t_kl_trajectory_values = 0
        t_pred_no_gt = 0
        t_gt_no_pred = 0
        t_gt_pred_f1 = 0
        kl_trajectory_values_sum = 0
        kl_trajectory_auc_sum = 0
        tvd_sum = 0
        for i in range(len(all_len)):
            if all_len[i] == 0:
                continue
            mini_prob = np.amin(all_diags[i, 0:all_len[i]], axis=0)
            maxi_prob = np.amax(all_diags[i, 0:all_len[i]], axis=0)    
            tvd_sum += np.absolute(maxi_prob - mini_prob).mean()
            kl_trajectory_values = kl_trajectory_score(explore_score[i, 0:all_len[i]], confirm_score[i, 0:all_len[i]])
            kl_trajectory_values_sum += np.mean(kl_trajectory_values)
            kl_trajectory_auc_sum += kl_trajectory_auc(explore_score[i, 0:all_len[i]], confirm_score[i, 0:all_len[i]])
            t_explore_score = t_explore_score + np.array(get_average_state_from_percent(explore_score[i, 0:all_len[i]], percent=p, end_percent=None))
            t_succesive_explore_score = t_succesive_explore_score + np.array(get_average_state_from_percent(succesive_explore_score[i, 0:all_len[i]-1], percent=p, end_percent=None))
            t_confirm_score = t_confirm_score + np.array(get_average_state_from_percent(confirm_score[i, 0:all_len[i]], percent=p, end_percent=None))
            t_kl_trajectory_values = t_kl_trajectory_values + np.array(get_average_state_from_percent(kl_trajectory_values, percent=p, end_percent=None))
            t_pred_no_gt = t_pred_no_gt + np.array(get_average_state_from_percent(pred_no_gt[i, 0:all_len[i]], percent=p, end_percent=None))
            t_gt_no_pred = t_gt_no_pred + np.array(get_average_state_from_percent(gt_no_pred[i, 0:all_len[i]], percent=p, end_percent=None))
            t_gt_pred_f1 = t_gt_pred_f1 + np.array(get_average_state_from_percent(gt_pred_f1[i, 0:all_len[i]], percent=p, end_percent=None))
        
        tvd_sum /= max(1, len(all_len))
        kl_trajectory_values_sum /= max(1, len(all_len))
        kl_trajectory_auc_sum /= max(1, len(all_len))
        t_explore_score /= max(1, len(all_len))
        t_succesive_explore_score /= max(1, len(all_len))
        t_confirm_score /= max(1, len(all_len))
        t_kl_trajectory_values /= max(1, len(all_len))
        t_pred_no_gt /= max(1, len(all_len))
        t_gt_no_pred /= max(1, len(all_len))
        t_gt_pred_f1 /= max(1, len(all_len))

        result["TVD"] = tvd_sum
        result["TrajScore"] = kl_trajectory_values_sum
        result["AUCTraj"] = kl_trajectory_auc_sum
        result["PlotData.x"] = np.array(p)
        result["PlotData.Exploration"] = t_explore_score
        result["PlotData.SuccessiveExploration"] = t_succesive_explore_score
        result["PlotData.Confirmation"] = t_confirm_score
        result["PlotData.Trajectory"] = t_kl_trajectory_values
        result["PlotData.SevF1"] = t_gt_pred_f1
        result["PlotData.SevPrecOut"] = t_pred_no_gt
        result["PlotData.SevRecIn"] = t_gt_no_pred
        
        return result
