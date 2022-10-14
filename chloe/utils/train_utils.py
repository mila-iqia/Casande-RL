import copy
import glob
import json
import os
import os.path as osp
import pprint

import numpy as np
import orion.client.cli as cli
import torch
import yaml
from orion.client import report_results
from rlpyt.samplers.collections import TrajInfo
from rlpyt.utils.logging import logger

from chloe.utils.dist_metric import (
    dist_accuracy,
    dist_js_div,
    dist_ncg,
    dist_ndcg,
    kl_confirm_score,
    kl_explore_score,
    kl_trajectory_auc,
    numpy_get_severe_pathos_inout_ratio,
    numpy_softmax,
)
from chloe.utils.runner_utils import EarlyStoppingError, PerformanceMixin
from chloe.utils.tensor_utils import _clamp_utils, _negate_tensor

RUN_ID_FILE_NAME = "exp_run_id.json"


class SimPaTrajInfo(TrajInfo):
    """TrajInfo class for use with SimPa Env.

    It overrides `rlpyt.samplers.collections.TrajInfo` by collecting
    information regarding:
       - repeated actions
       - number of inquired antecedents/symptoms
       - number of relevant antecedents/symptoms
       - number of experienced symptoms
       - number of experienced antecedents
       - simulated pathologies
       - predicted pathologies
       - pathology severity
       - interaction session timestep
       - agent predicted distribution
    """

    def __init__(self, **kwargs):
        """Instantiates a `SimPaTrajInfo` object.
        """
        super().__init__(**kwargs)
        self.NumRepeatedSymptoms = 0
        self.NumInquiredAntecedents = 0
        self.NumInquiredSymptoms = 0
        self.NumRelevantSymptoms = 0
        self.NumRelevantAntecedents = 0
        self.PrecisionRelevantSymptoms = 0
        self.PrecisionRelevantAntecedents = 0
        self.RecallRelevantSymptoms = 0
        self.RecallRelevantAntecedents = 0
        self.AvgInfoGainOnIrrelevancy = 0
        self.NdcgDistPred = 0
        self.NdcgBasedAggScore = 0
        self.NcgDistPred = 0
        self.NcgBasedAggScore = 0
        self.AvgPrecDistPred = 0
        self.AvgPrecAggScore = 0
        self.AvgPrecFullDistPred = 0
        self.AvgPrecFullAggScore = 0
        self.ExplConfAUC = 0
        self.ExplConfSortedAUC = 0
        self.ExplConfAUCAggScore = 0
        self.ExplConfSortedAUCAggScore = 0
        self.SevOut = 0
        self.SevIn = 0
        self.SevF1 = 0
        self._num_experienced_symptoms = 0
        self._num_experienced_antecedents = 0
        self._num_evidences = 0
        self._metric_y_true = 0
        self._tensor_y_true = None
        self._metric_diffential_indices = None
        self._tensor_diffential_indices = None
        self._metric_diffential_probas = None
        self._tensor_diffential_probas = None
        self._metric_evidence = None
        self._tensor_evidence = None
        self._metric_y_pred = 0
        self._sim_severity = 0
        self._sim_timestep = 0
        self._tensor_sim_timestep = None
        self._first_dist_info = None
        self._prev_dist_info = None
        self._dist_info = None
        self._kl_explore = []
        self._kl_confirm = []
        self._info_gain_on_irrelevancy = 0
        self._num_irrelevancy = 0
        self._num_evidenced_inquiries = 0
        self._sim_age = -1
        self._sim_sex = -1
        self._sim_race = -1
        self._sim_geo = -1

    def _update_attr_key(self, key, val, cur_discount):
        """Update the value of the provided attribute.

        This function updates the attribute by summing up its
        current value with the provided one. Additionnaly, the
        function updates a discounted version of the attribute
        where the summation is modulated by the provided discount
        value.

        Parameters
        ----------
        key: str
            the key associated to the attribute to update.
        val: float
            the value to update the attribute with.
        cur_discount: float
            the current value of the discount factor at the update step.

        Returns
        -------
        None

        """
        setattr(self, key, getattr(self, key, 0.0) + val)
        setattr(
            self,
            "Discounted" + key,
            getattr(self, "Discounted" + key, 0.0) + (cur_discount * val),
        )

    def _define_tensors(self):
        """Defines the tensors for computing auxiliary rewards (reward shaping).
        """
        with torch.no_grad():
            if self._tensor_y_true is None:
                self._tensor_y_true = torch.tensor([self._metric_y_true])
            if self._tensor_sim_timestep is None:
                self._tensor_sim_timestep = torch.tensor([self._sim_timestep])
            else:
                self._tensor_sim_timestep += 1
            if (
                self._tensor_diffential_indices is None
                and self._metric_diffential_indices is not None
            ):
                self._tensor_diffential_indices = torch.from_numpy(
                    self._metric_diffential_indices
                ).unsqueeze(0)
            if (
                self._tensor_diffential_probas is None
                and self._metric_diffential_probas is not None
            ):
                self._tensor_diffential_probas = torch.from_numpy(
                    self._metric_diffential_probas
                ).unsqueeze(0)
            self._tensor_evidence = torch.tensor([self._metric_evidence])

    def _compute_reward_shaping(self, cur_discount):
        """Computes the intermediate auxiliary reward at each step of the episode.

        Parameters
        ----------
        cur_discount: float
            the current value of the discount factor at the update step.

        Returns
        -------
        result: float
            the computed reward.

        """
        reward_shaping_flag = self._aux_reward_info.get("reward_shaping_flag", False)
        coef = self._aux_reward_info.get("reward_shaping_coef", 1.0)
        aux_rs = 0.0
        aux_dict = {}
        result = 0.0
        keys = getattr(SimPaTrajInfo, "_aux_rs_keys", [])
        if reward_shaping_flag and (self._prev_dist_info is not None) and (coef != 0):
            with torch.no_grad():
                dist_info = (
                    torch.from_numpy(self._dist_info)
                    if isinstance(self._dist_info, np.ndarray)
                    else self._dist_info
                )
                prev_dist_info = (
                    torch.from_numpy(self._prev_dist_info)
                    if isinstance(self._prev_dist_info, np.ndarray)
                    else self._prev_dist_info
                )
                aux_rs, aux_dict = self._aux_reward_info[
                    "reward_shaping_factory"
                ].evaluate(
                    self._aux_reward_info["reward_shaping_func"],
                    dist_info.unsqueeze(0),
                    prev_dist_info.unsqueeze(0),
                    self._tensor_y_true,
                    self._tensor_diffential_indices,
                    self._tensor_diffential_probas,
                    self._tensor_evidence,
                    self._discount,
                    self._aux_reward_info["patho_severity"],
                    self._tensor_sim_timestep,
                    **self._aux_reward_info["reward_shaping_kwargs"],
                )
                aux_rs = _clamp_utils(
                    aux_rs,
                    self._aux_reward_info["reward_shaping_min"],
                    self._aux_reward_info["reward_shaping_max"],
                ).item()

                if aux_dict is None:
                    aux_dict = {}
                for k in aux_dict.keys():
                    aux_dict[k] = (
                        aux_dict[k].item()
                        if isinstance(aux_dict[k], torch.Tensor)
                        else aux_dict[k]
                    )

                aux_rs_keys = [""] + list(aux_dict.keys())
                diff = set(aux_rs_keys) - set(keys)
                if diff:
                    keys += list(diff)
                    setattr(SimPaTrajInfo, "_aux_rs_keys", keys)

                result = coef * aux_rs

        for k in keys:
            val = aux_rs if k == "" else aux_dict.get(k, 0.0)
            # saving
            self._update_attr_key("Auxiliary" + k + "Return", coef * val, cur_discount)

        return result

    def _compute_clf_reward(self, cur_discount, done):
        """Computes the classifier auxiliary reward at the end of the episode.

        Parameters
        ----------
        cur_discount: float
            the current value of the discount factor at the update step.
        done: bool
            a flag indicating the end of the episode.

        Returns
        -------
        result: float
            the computed reward.

        """
        clf_reward_flag = self._aux_reward_info.get("clf_reward_flag", False)
        coef = self._aux_reward_info.get("clf_reward_coef", 1.0)
        clf_rs = 0.0
        clf_dict = {}
        result = 0.0
        keys = getattr(SimPaTrajInfo, "_clf_rs_keys", [])
        if clf_reward_flag and done and (coef != 0) and (self._dist_info is not None):
            with torch.no_grad():
                dist_info = (
                    torch.from_numpy(self._dist_info)
                    if isinstance(self._dist_info, np.ndarray)
                    else self._dist_info
                )
                dist_info = dist_info.unsqueeze(0).transpose(1, -1)
                clf_rs, clf_dict = self._aux_reward_info["clf_reward_factory"].evaluate(
                    self._aux_reward_info["clf_reward_func"],
                    dist_info,
                    self._tensor_y_true,
                    self._tensor_diffential_indices,
                    self._tensor_diffential_probas,
                    reduction="none",
                    severity=self._aux_reward_info["patho_severity"],
                    timestep=self._tensor_sim_timestep,
                    **self._aux_reward_info["clf_reward_kwargs"],
                )
                clf_rs = _negate_tensor(clf_rs)
                clf_dict = _negate_tensor(clf_dict)

                clf_rs = _clamp_utils(
                    clf_rs,
                    self._aux_reward_info["clf_reward_min"],
                    self._aux_reward_info["clf_reward_max"],
                ).item()

                if clf_dict is None:
                    clf_dict = {}
                for k in clf_dict.keys():
                    clf_dict[k] = (
                        clf_dict[k].item()
                        if isinstance(clf_dict[k], torch.Tensor)
                        else clf_dict[k]
                    )

                clf_rs_keys = [""] + list(clf_dict.keys())
                diff = set(clf_rs_keys) - set(keys)
                if diff:
                    keys += list(diff)
                    setattr(SimPaTrajInfo, "_clf_rs_keys", keys)

                result = coef * clf_rs

        for k in keys:
            val = clf_rs if k == "" else clf_dict.get(k, 0.0)
            # saving
            self._update_attr_key("Classifier" + k + "Return", coef * val, cur_discount)

        return result

    def _compute_auxiliary_rewards(self, reward, done):
        """Computes and stores auxiliary rewards.

        Parameters
        ----------
        reward: float
            the reward received from the environment at the current step.
        done: bool
            a flag indicating the end of the episode.

        Returns
        -------
        None

        """
        flag_str = "traj_auxiliary_reward_flag"
        if self._aux_reward_info and self._aux_reward_info.get(flag_str, False):
            env_coef = self._aux_reward_info.get("env_reward_coef", 1.0)
            self._define_tensors()
            cur_discount = self._cur_discount / self._discount
            # compute auxiliary rewards (intermediate auxiliary reward and clf reward)
            aux_rs = self._compute_reward_shaping(cur_discount)
            clf_rs = self._compute_clf_reward(cur_discount, done)
            # weighted env reward
            env_rew = env_coef * reward
            # saving
            self._update_attr_key("EnvReturn", env_rew, cur_discount)
            self._update_attr_key(
                "OptimizedReturn", env_rew + aux_rs + clf_rs, cur_discount
            )

    def _compute_antecedent_symtoms_stats(self, done):
        """Computes and stores relevancy related metrics on data acquisition.

        Parameters
        ----------
        done: bool
            a flag indicating the end of the episode.

        Returns
        -------
        None

        """
        if done:
            assert self.NumRelevantSymptoms <= self._num_experienced_symptoms
            assert self.NumRelevantAntecedents <= self._num_experienced_antecedents
            self.PrecisionRelevantSymptoms = (
                1.0
                if self.NumInquiredSymptoms == 0
                else self.NumRelevantSymptoms / self.NumInquiredSymptoms
            )
            self.PrecisionRelevantAntecedents = (
                1.0
                if self.NumInquiredAntecedents == 0
                else self.NumRelevantAntecedents / self.NumInquiredAntecedents
            )
            self.RecallRelevantSymptoms = (
                1.0
                if self._num_experienced_symptoms == 0
                else self.NumRelevantSymptoms / self._num_experienced_symptoms
            )
            self.RecallRelevantAntecedents = (
                1.0
                if self._num_experienced_antecedents == 0
                else self.NumRelevantAntecedents / self._num_experienced_antecedents
            )
            self.AvgInfoGainOnIrrelevancy = (
                1.0
                if self._num_irrelevancy == 0
                else self._info_gain_on_irrelevancy / self._num_irrelevancy
            )

    def _compute_distribution_stats(self, done):
        """Computes and stores relevancy related metrics on data prediction.

        Parameters
        ----------
        done: bool
            a flag indicating the end of the episode.

        Returns
        -------
        None

        """
        if done and self._dist_info is not None:
            pred = (
                self._dist_info.numpy()
                if isinstance(self._dist_info, torch.Tensor)
                else self._dist_info
            )
            self.NdcgDistPred = dist_ndcg(
                pred,
                self._metric_y_true,
                self._metric_diffential_indices,
                self._metric_diffential_probas,
                self._topk,
                ignore_index=-1,
            )
            self.NcgDistPred = dist_ncg(
                pred,
                self._metric_y_true,
                self._metric_diffential_indices,
                self._metric_diffential_probas,
                self._topk,
                ignore_index=-1,
            )
            self.AvgPrecDistPred = dist_accuracy(
                pred,
                self._metric_y_true,
                self._metric_diffential_indices,
                self._metric_diffential_probas,
                self._topk,
                restrict_to_k=True,
                ignore_index=-1,
            )
            self.AvgPrecFullDistPred = dist_accuracy(
                pred,
                self._metric_y_true,
                self._metric_diffential_indices,
                self._metric_diffential_probas,
                self._topk,
                restrict_to_k=False,
                ignore_index=-1,
            )

    def _compute_trajectory_score(self, done, proba):
        """Computes and stores the score of the whole trajectory/episode.

        Parameters
        ----------
        done: bool
            a flag indicating the end of the episode.
        proba: np.ndarray
            the current proba distribution outputed by the agent.

        Returns
        -------
        None

        """
        if done:
            # if kl_explore and kl_confirm is not empty
            if len(self._kl_explore) > 0 and len(self._kl_confirm) > 0:
                self.ExplConfAUC = kl_trajectory_auc(
                    self._kl_explore, self._kl_confirm, "none"
                )
                self.ExplConfSortedAUC = kl_trajectory_auc(
                    self._kl_explore, self._kl_confirm, "sort"
                )
            # compute severity stats
            severity = self._aux_reward_info["patho_severity"]
            threshlod = self._aux_reward_info.get("severity_threshold", 3)
            severe_indices = (
                None
                if severity is None
                else np.where(severity.cpu().numpy() < threshlod)[0]
            )
            sev_out, sev_in = (
                (0.0, 0.0)
                if proba is None
                else numpy_get_severe_pathos_inout_ratio(
                    proba.reshape((1, -1)),
                    self._metric_y_true,
                    self._metric_diffential_indices,
                    self._metric_diffential_probas,
                    severe_indices,
                    diff_proba_threshold=0.01,
                    front_broadcast_flag=True,
                )
            )
            sev_f1 = (2 * sev_out * sev_in) / (sev_out + sev_in + 1e-10)
            self.SevOut = sev_out
            self.SevIn = sev_in
            self.SevF1 = sev_f1
            # compute the sum of others components except the 1st one
            tmp = (
                (self.RecallRelevantSymptoms * self._eval_coeffs[1])
                + (self.RecallRelevantAntecedents * self._eval_coeffs[2])
                + (self.AvgInfoGainOnIrrelevancy * self._eval_coeffs[3])
            )
            # get the sum of all the coefficients for normalization purposes
            tot = sum(self._eval_coeffs)
            # aggregation based on ndcg ranking metric
            self.NdcgBasedAggScore = (
                (self.NdcgDistPred * self._eval_coeffs[0]) + tmp
            ) / tot
            # aggregation based on ncg ranking metric
            self.NcgBasedAggScore = (
                (self.NcgDistPred * self._eval_coeffs[0]) + tmp
            ) / tot
            # aggregation based on avg precision ranking metric
            # (as ncg but with boolean relevance)
            self.AvgPrecAggScore = (
                (self.AvgPrecDistPred * self._eval_coeffs[0]) + tmp
            ) / tot
            # aggregation based on avg precision ranking metric
            # with full differential (as ncg but with boolean relevance)
            self.AvgPrecFullAggScore = (
                (self.AvgPrecFullDistPred * self._eval_coeffs[0]) + tmp
            ) / tot
            # aggregation based on ExplConfAUC
            self.ExplConfAUCAggScore = (
                (self.ExplConfAUC * self._eval_coeffs[0]) + tmp
            ) / tot
            # aggregation based on ExplConfAUC
            self.ExplConfSortedAUCAggScore = (
                (self.ExplConfSortedAUC * self._eval_coeffs[0]) + tmp
            ) / tot

    def _get_dist_infos(self, agent_info, env_info):
        """Get and stores the data related to the pathology prediction by the agent.

        Parameters
        ----------
        agent_info: namedtuple
            data structure containing additionnal info from the agent.
        env_info: namedtuple
            data structure containing additionnal info from the environment.

        Returns
        -------
        dist_info: tensor
            the predicted data.

        """
        result = getattr(agent_info, "dist_info", None)
        if result is None:
            num_pathos = getattr(env_info, "sim_total_num_pathos", 0)
            q = self._get_q_values_infos(agent_info)
            if q is not None:
                result = q[..., -num_pathos:]
        elif hasattr(result, "prob"):
            num_pathos = getattr(env_info, "sim_total_num_pathos", 0)
            result = getattr(result, "prob", None)
            result = None if result is None else result[..., -num_pathos:]
        return result

    def _get_q_values_infos(self, agent_info):
        """Get and stores the data related to the Q-values prediction by the agent.

        Parameters
        ----------
        agent_info: namedtuple
            data structure containing additionnal info from the agent.

        Returns
        -------
        q_values: tensor
            the predicted Q-values.

        """
        key = "q" if self._p_z is None else "p"
        q = getattr(agent_info, key, None)
        if self._p_z is not None and q is not None:
            q = (
                torch.tensordot(q, self._p_z, dims=1)
                if isinstance(q, torch.Tensor)
                else np.dot(q, self._p_z.numpy())
            )
        return q

    def _compute_kl_scores(self, proba):
        """Computes the kl_confirm and kl_explore score.

        Parameters
        ----------
        proba: np.ndarray
            the current proba distribution outputed by the agent.
        """
        self._kl_explore.append(
            0.0
            if (self._first_dist_info is None or proba is None)
            else kl_explore_score(proba.reshape((1, -1)), self._first_dist_info)[0]
        )
        self._kl_confirm.append(
            0.0
            if proba is None
            else kl_confirm_score(
                proba.reshape((1, -1)),
                self._metric_y_true,
                self._metric_diffential_indices,
                self._metric_diffential_probas,
            )[0]
        )

    def step(self, observation, action, reward, done, agent_info, env_info):
        """Computes the trajectory info at each step of the episode.

        Parameters
        ----------
        observation: np.ndarray
            the observation data as provided to the agent.
        action: int
            the undertaken action at the current step.
        reward: float
            the reward received at the current step.
        done: bool
            a flag indicating the end of the episode.
        agent_info: namedtuple
            data structure containing additionnal info from the agent.
        env_info: namedtuple
            data structure containing additionnal info from the environment.

        Returns
        -------
        None

        """
        super().step(observation, action, reward, done, agent_info, env_info)
        self._prev_dist_info = self._dist_info
        self._dist_info = self._get_dist_infos(agent_info, env_info)
        current_proba_dist = (
            None
            if self._dist_info is None
            else numpy_softmax(
                self._dist_info.cpu().numpy()
                if isinstance(self._dist_info, torch.Tensor)
                else self._dist_info
            )
        )
        self._first_dist_info = (
            self._first_dist_info
            if self.Length > 1
            else (
                None if current_proba_dist is None else current_proba_dist.reshape((-1))
            )
        )
        if (not done) or (getattr(env_info, "diagnostic", -1) == -1):
            repeated = getattr(env_info, "is_repeated_action", 0)
            self.NumRepeatedSymptoms += repeated
            antecedent = getattr(env_info, "is_antecedent_action", 0)
            relevancy = getattr(env_info, "is_relevant_action", 0)
            is_evidence = getattr(env_info, "sim_evidence", 0)
            self.NumInquiredAntecedents += 1 if antecedent != 0 else 0
            self.NumInquiredSymptoms += 1 if antecedent == 0 else 0
            self.NumRelevantSymptoms += (
                1 if relevancy != 0 and antecedent == 0 and repeated == 0 else 0
            )
            self.NumRelevantAntecedents += (
                1 if relevancy != 0 and antecedent != 0 and repeated == 0 else 0
            )
            self._num_irrelevancy += 1 if is_evidence == 0 or repeated != 0 else 0
            is_null_dist = self._dist_info is None or self._prev_dist_info is None
            self._info_gain_on_irrelevancy += (
                0
                if ((is_evidence != 0 and repeated == 0) or is_null_dist)
                else dist_js_div(
                    self._prev_dist_info.numpy()
                    if isinstance(self._prev_dist_info, torch.Tensor)
                    else self._prev_dist_info,
                    self._dist_info.numpy()
                    if isinstance(self._dist_info, torch.Tensor)
                    else self._dist_info,
                )
            )
        self._num_experienced_symptoms = getattr(env_info, "sim_num_symptoms", 0)
        self._num_experienced_antecedents = getattr(env_info, "sim_num_antecedents", 0)
        self._num_evidences = getattr(env_info, "sim_num_evidences", 0)
        self._metric_y_true = getattr(env_info, "sim_patho", 0)
        self._metric_y_pred = getattr(env_info, "diagnostic", 0)
        self._sim_severity = getattr(env_info, "sim_severity", 0)
        self._sim_timestep = getattr(env_info, "sim_timestep", 0)
        self._metric_evidence = getattr(env_info, "sim_evidence", 0)
        self._sim_age = getattr(env_info, "sim_age", -1)
        self._sim_sex = getattr(env_info, "sim_sex", -1)
        self._sim_race = getattr(env_info, "sim_race", -1)
        self._sim_geo = getattr(env_info, "sim_geo", -1)
        self._num_evidenced_inquiries += self._metric_evidence
        self._metric_diffential_indices = getattr(
            env_info, "sim_differential_indices", None
        )
        self._metric_diffential_probas = getattr(
            env_info, "sim_differential_probas", None
        )
        self._compute_kl_scores(current_proba_dist)
        self._compute_antecedent_symtoms_stats(done)
        self._compute_distribution_stats(done)
        self._compute_auxiliary_rewards(reward, done)
        self._compute_trajectory_score(done, current_proba_dist)


class SimPaTrajEvalInfo(SimPaTrajInfo):
    """TrajInfo class for use when evaluating SimPa Env.

    It overrides `SimPaTrajInfo` by collecting information regarding:
       - relevant actions
       - first symptoms
       - simulated patient data.
       - distribution info along the trajectory
    """

    def __init__(self, **kwargs):
        """Instantiates a `SimPaTrajEvalInfo` object.
        """
        super().__init__(**kwargs)
        self._inquired_symptoms = []
        self._relevant_actions = []
        self._repeated_actions = []
        self._atcd_actions = []
        self._all_proba_distributions = []
        self._all_q_values = []
        self._first_symptom = -1
        self._gathered_infos = None
        self._simulated_patient = None
        self._all_rewards_data = {}

    def _update_attr_key(self, key, val, cur_discount):
        """Update the value of the provided attribute.

        This function updates the attribute by summing up its
        current value with the provided one. Additionnaly, the
        function updates a discounted version of the attribute
        where the summation is modulated by the provided discount
        value.

        Parameters
        ----------
        key: str
            the key associated to the attribute to update.
        val: float
            the value to update the attribute with.
        cur_discount: float
            the current value of the discount factor at the update step.

        Returns
        -------
        None

        """
        super()._update_attr_key(key, val, cur_discount)
        if key not in self._all_rewards_data:
            self._all_rewards_data[key] = []
        self._all_rewards_data[key].append(val)

    def step(self, observation, action, reward, done, agent_info, env_info):
        """Computes the trajectory info at each step of the episode.

        Parameters
        ----------
        observation: np.ndarray
            the observation data as provided to the agent
        action: int
            the undertaken action at the current step.
        reward: float
            the reward received at the current step.
        done: bool
            a flag indicating the end of the episode.
        agent_info: namedtuple
            data structure containing additionnal info from the agent.
        env_info: namedtuple
            data structure containing additionnal info from the environment.

        Returns
        -------
        None

        """
        super().step(observation, action, reward, done, agent_info, env_info)
        if getattr(env_info, "current_symptom", -1) != -1:
            self._inquired_symptoms.append(getattr(env_info, "current_symptom"))
        self._relevant_actions.append(getattr(env_info, "is_relevant_action", 0))
        self._repeated_actions.append(getattr(env_info, "is_repeated_action", 0))
        self._atcd_actions.append(getattr(env_info, "is_antecedent_action", 0))
        self._first_symptom = getattr(env_info, "first_symptom", -1)
        self._gathered_infos = observation
        self._simulated_patient = getattr(env_info, "sim_patient", None)
        self._all_q_values.append(self._get_q_values_infos(agent_info))
        self._all_proba_distributions.append(self._dist_info)


def reload_snapshot(output, filename, start_from_scratch=False):
    """Method for loading a snapshot of the training process (checkpointing).

    Parameters
    ----------
    output: str
        the dir containing the logged snapshot
    filename: str
        the name of the file corresponding to the snapshot to be
        loaded.
    start_from_scratch: bool
        if True, will not load any existing saved model - even if present.
        Default: False

    Returns
    -------
    result: dict
        the resulting data containing related checkpinting infos.

    """
    result = {}
    saved_file = os.path.join(output, filename)
    if os.path.exists(saved_file):
        if start_from_scratch:
            if os.path.exists(saved_file):
                logger.log(
                    'saved model file "{}" already exists - but NOT loading it '
                    "(cause --start_from_scratch)".format(saved_file)
                )
        else:
            if os.path.exists(saved_file):
                logger.log(
                    'saved model file "{}" already exists - loading it'.format(
                        saved_file
                    )
                )
                result = torch.load(saved_file)

    elif os.path.exists(output):
        logger.log(
            "saved model file not found - but output folder exists already - keeping it"
        )
    else:
        logger.log("no saved model file found - nor output folder - creating it")
        os.makedirs(output)

    return result


def load_pretrained_snapshot(resume_snapshot, filename):
    """Method for loading a snapshot of the pretrained model.

    This function returns `resume_snapshot` if `filename` is None or empty.

    Parameters
    ----------
    resume_snapshot: dict
        snapshop related to resume operation.
    filename: str
        filename of the pretrained model snapshot of the to be loaded.

    Returns
    -------
    result: dict
        the resulting snapshot.

    """
    if not filename:
        return resume_snapshot
    if resume_snapshot:
        raise ValueError(
            "Trying to load a pretrained model snapshot while a resume snapshot "
            "exist and has previoulsy been loaded. There is an ambiguity. Please "
            "consider either not loading the resume snapshot or not loading the "
            "pretrained model snapshot"
        )

    result = {}
    if os.path.exists(filename):
        result = torch.load(filename)
    else:
        raise ValueError(f"the specified file ({filename}) does not exist")

    return result


def train(runner):
    """Method for training the RL agent.

    Parameters
    ----------
    runner: Runner
        the runner to be used for training.

    Returns
    -------
    None

    """
    try:
        try:
            runner.train()
        except EarlyStoppingError:
            # this is not an error. It is just an exception informing us
            # that the process ended because of early stopping.
            pass
        if isinstance(runner, PerformanceMixin):
            best_dev_metric = runner.get_final_performance()
        else:
            best_dev_metric = None
    except RuntimeError as err:
        if cli.IS_ORION_ON and "CUDA out of memory" in str(err):
            logger.log(err)
            if isinstance(runner, PerformanceMixin):
                logger.log(
                    "model was out of memory - assigning a bad score to tell Orion"
                    " to avoidtoo big model"
                )
                best_dev_metric = -999
            else:
                best_dev_metric = None
        else:
            raise err

    if best_dev_metric is not None:
        report_results(
            [
                dict(
                    name="dev_metric",
                    type="objective",
                    # note the minus - cause orion is always trying
                    # to minimize (cit. from the guide)
                    value=-float(best_dev_metric),
                )
            ]
        )


def save_run_ids(output_dir, mlflow_run_id, logging_run_id):
    """Method for saving the id of a running experiment.

    Parameters
    ----------
    output_dir: str
        the provided ouput dir.
    mlflow_run_id: str
        the mlflow id of a running experiment.
    logging_run_id: int
        the logging id of a running experiment.

    Returns
    -------
    None

    """
    params_log_file = osp.join(output_dir, RUN_ID_FILE_NAME)
    log_params = dict()
    log_params["mlflow_run_id"] = mlflow_run_id
    log_params["logging_run_id"] = logging_run_id
    with open(params_log_file, "w") as f:
        json.dump(log_params, f)


def check_run_id_file_existence(output_dir):
    """Method for checking if a run_id file exist in the provided dir.

    Parameters
    ----------
    output_dir: str
        the provided ouput dir.

    Returns
    -------
    result: bool
        True if the file can be found in the provided dir.
        False otherwise.

    """
    params_log_file = osp.join(output_dir, RUN_ID_FILE_NAME)
    return osp.exists(params_log_file)


def load_run_ids(output_dir):
    """Method for loading the run_id from a provided experiment dir.

    Parameters
    ----------
    output_dir: str
        the provided ouput dir.

    Returns
    -------
    mlflow_run_id: str
        the mlflow id loaded from the run_id file in the provided dir.
    logging_run_id: int
        the logging id loaded from the run_id file in the provided dir.

    """
    params_log_file = osp.join(output_dir, RUN_ID_FILE_NAME)
    with open(params_log_file) as f:
        data = json.load(f)
        mlflow_run_id = data["mlflow_run_id"]
        logging_run_id = data["logging_run_id"]
    return mlflow_run_id, logging_run_id


def update_config_and_setup_exp_folder(args, cfg, uid):
    """Update the configs based on args flags and setup the experiment folder as needed.
    Parameters
    ----------
    args : NameSpace
    cfg : dict
        Raw config loaded directly from the config file.
    uid : str
        Unique id for the experiment run.
    Returns
    -------
    cfg : dict
        Config dictionary.
    """

    cfg["mlflow_uid"] = cfg.get("mlflow_uid", uid)

    cfg["config_uid"] = cfg.get("config_uid", uid)

    # Checkpointing
    if "experiment_folder" not in cfg:
        # Add experiment_folder to cfg with unique identifier
        exp_subfolder = os.path.join(cfg["exp_name"], cfg["config_uid"])
        cfg["experiment_folder"] = os.path.join(args.output, exp_subfolder)

    run_id = f"run_{args.run_ID}"
    exp_folder = os.path.join(cfg["experiment_folder"], run_id)
    if os.path.exists(exp_folder):
        #  if check_run_id_file_existence(exp_folder):
        #      mlflow_uid, logging_run_id = load_run_ids(exp_folder)
        #      assert mlflow_uid == uid
        #      assert args.run_ID == logging_run_id
        start_from_scratch = "start_from_scratch" in args and args.start_from_scratch
        if start_from_scratch:
            file_names = glob.glob(os.path.join(exp_folder, "*.pkl"))
            file_names += [
                os.path.join(exp_folder, filea)
                for filea in [
                    "cfg.yml",
                    "git_diff.txt",
                    "params.json",
                    "progress.csv",
                    "debug.log",
                ]
            ]
            for file_name in file_names:
                if os.path.isfile(file_name):
                    os.remove(file_name)
    else:
        os.makedirs(exp_folder)

    # Make a copy to be able to delete/modify keys without impacting cfg
    save_cfg = copy.deepcopy(cfg)

    # Overwrite args.config file for checkpointing purpose
    if "config" in args:
        with open(args.config, "w") as f:
            yaml.dump(save_cfg, f)
    # Saving copy to experiment folder
    with open(os.path.join(exp_folder, "cfg.yml"), "w") as f:
        yaml.dump(save_cfg, f)

    # Print config
    print("\nUsing config:")
    pprint.pprint(cfg)

    return cfg
