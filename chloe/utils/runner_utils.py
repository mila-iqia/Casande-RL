import collections
import os.path as osp

import psutil
import torch
from rlpyt.runners.async_rl import AsyncRl, AsyncRlBase, AsyncRlEval
from rlpyt.runners.minibatch_rl import MinibatchRl, MinibatchRlBase, MinibatchRlEval
from rlpyt.runners.sync_rl import SyncRl, SyncRlEval
from rlpyt.utils.launching.affinity import make_affinity
from rlpyt.utils.logging import logger
from rlpyt.utils.prog_bar import ProgBarCounter

from chloe.evaluator.batchEvaluator import evaluate as batch_evaluate
from chloe.utils.eval_utils import MetricFactory
from chloe.utils.sampler_utils import is_async_sampler, is_aternating_sampler
from chloe.utils.scheduler_utils import numpy_sigmoid_scheduler

BEST_MODEL_NAME = "best.pkl"


class EarlyStoppingError(Exception):
    """Exception Class to be used Early stopping is performed.
    """

    pass


class PerformanceMixin:
    """Extends the runner functionality to keep track of the best training performance.

    Mixin class to extend runner functionality to keep track of the best
    performance over training as well as keeping the average performance
    over the last `windows size` evaluations. Useful for resuming
    experiments as well as dealing with hyper-opt framework such as orion.
    """

    def set_resume_params(
        self,
        resume_info={},
        metrics=None,
        perf_window_size=1,
        patience=None,
        topk=1,
        eval_coeffs=None,
        perf_metric="Reward",
    ):
        """Method for setting up the resuming parameter for a running experiment.

        Parameters
        ----------
        resume_info: dict
            dictionary containing resuming infos. Default: dict.
        metrics: list, None
            list of metrics to monitor during training.
        perf_window_size: int
            the number of last evaluation performace to consider.
            Default: 1
        patience: int
            the number of iterations we may tolerate performance degradiation.
            Useful for early stopping. if None, ealy_stopping is disable.
            Default: None
        topk: int
            the number of top predicted pathologies to be considered during evaluation.
            Default: 1
        eval_coeffs: list
            the coefficients to be used to weight each component of the aggregated
            evaluation metrics. If None, the components are equally weighted.
            These components are defined in the following order:
            - Differential Diagnosis score
            - Symptom Discovery score
            - Risk Factor Discovery score
            - Negative Response Pertinence score
            Default: None
        perf_metric: str
            the metric used to monitor the evaluation. if None, use the reward.
            Default: Reward

        Returns
        -------
        None

        """
        self.init_itr = resume_info.get("itr", 0)
        self.init_sampler_itr = resume_info.get("sampler_itr", 0)
        self.best_eval_performance = resume_info.get("best_eval_performance", None)
        self.eval_metrics = metrics
        self.max_patience = patience
        self.perf_metric = perf_metric
        self.topk = topk
        self.eval_coeffs = [1] * 4 if eval_coeffs is None else eval_coeffs
        assert len(self.eval_coeffs) >= 4, "the coefficient's number must be at least 4"
        self.remaining_patience = patience
        self.metric_factory = MetricFactory()
        self.init_cum_updates = resume_info.get("cum_updates", 0)
        self.init_cum_trajectories = resume_info.get("cum_completed_trajs", 0)
        self.perf_window_size = perf_window_size
        self.traj_auxiliary_reward_flag = False
        self.last_perf_windows = collections.deque(maxlen=perf_window_size)
        init_best_perf = self.best_eval_performance
        if init_best_perf is not None:
            self.last_perf_windows.append(init_best_perf)

    def set_flag_for_logging_trajectory_auxiliary_reward(self, flag):
        """Method to define whether to log or not auxiliary rewards in trajectory infos.

        Parameters
        ----------
        flag: boolean
            whether to log the auxiliary reward (if computed) or not.

        Returns
        -------
        None

        """
        self.traj_auxiliary_reward_flag = flag

    def set_batch_eval_context(self, batch_env, eval_batch_size):
        """Set context for batch evaluation during training.

        Parameters
        ----------
        batch_env: env
            batch environment
        eval_batch_size: int
            batch size for evaluation

        Returns
        -------
        None

        """
        if batch_env is not None:
            assert eval_batch_size is not None
        self.batch_env = batch_env
        self.eval_batch_size = eval_batch_size
        self._setModelVMinMax = False

    def get_traj_info_kwargs(self):
        """Method for defining the variable arguments for trajectory info intances.

        This function pre-defines any TrajInfo attributes needed from elsewhere e.g.
        algorithm discount factor.

        This function overrides the existing function by providing additional parameters
        for auxiliary rewards involved in the reward shaping procedure.

        Parameters
        ----------

        Returns
        -------
        result: dict
            the dictionary of variable argument for the trajectory info intances.

        """
        result = super().get_traj_info_kwargs()
        patho_severity = self.agent.model_kwargs.get("patho_severity", None)
        if patho_severity is not None:
            patho_severity = torch.tensor(patho_severity).float()
        v_min = getattr(self.algo, "V_min", None)
        v_max = getattr(self.algo, "V_max", None)
        atoms = getattr(self.agent, "n_atoms", None)
        if not (v_min is None or v_max is None or atoms is None):
            p_z = torch.linspace(v_min, v_max, atoms)
        else:
            p_z = None
        aux_reward_info = dict(
            traj_auxiliary_reward_flag=self.traj_auxiliary_reward_flag,
            env_reward_coef=getattr(self.algo, "env_reward_coef", 1.0),
            clf_reward_coef=getattr(self.algo, "clf_reward_coef", 1.0),
            clf_reward_flag=getattr(self.algo, "clf_reward_flag", False),
            clf_reward_min=getattr(self.algo, "clf_reward_min", None),
            clf_reward_max=getattr(self.algo, "clf_reward_max", None),
            clf_reward_func=getattr(self.algo, "clf_reward_func", None),
            clf_reward_kwargs=getattr(self.algo, "clf_reward_kwargs", {}),
            clf_reward_factory=getattr(self.algo, "clf_loss_factory", None),
            reward_shaping_coef=getattr(self.algo, "reward_shaping_coef", 1.0),
            reward_shaping_flag=getattr(self.algo, "reward_shaping_flag", False),
            reward_shaping_min=getattr(self.algo, "reward_shaping_min", None),
            reward_shaping_max=getattr(self.algo, "reward_shaping_max", None),
            reward_shaping_func=getattr(self.algo, "reward_shaping_func", None),
            reward_shaping_kwargs=getattr(self.algo, "reward_shaping_kwargs", {}),
            reward_shaping_factory=getattr(self.algo, "reward_shaping_factory", None),
            patho_severity=patho_severity,
        )
        result["aux_reward_info"] = aux_reward_info
        result["topk"] = self.topk
        result["eval_coeffs"] = self.eval_coeffs
        result["p_z"] = p_z
        return result

    def initialize_logging(self):
        """Method for initializing the logging process of the runner instance.

        Parameters
        ----------

        Returns
        -------
        None

        """
        super().initialize_logging()
        if hasattr(self, "_last_itr"):
            self._last_itr = self.init_itr
        if hasattr(self, "_last_sampler_itr"):
            self._last_sampler_itr = self.init_sampler_itr
        if hasattr(self, "_last_update_counter"):
            self._last_update_counter = self.init_cum_updates
        if hasattr(self, "_cum_completed_trajs"):
            self._cum_completed_trajs = self.init_cum_trajectories
        if self.init_itr > 0:
            if not hasattr(self, "pbar"):
                self.pbar = ProgBarCounter(self.log_interval_itrs)

    def _log_infos(self, traj_infos=None):
        """Log trajectory info using the logger.

        Parameters
        ----------
        traj_infos: list, None
            list of trajectory infos to be logged. default: None

        Returns
        -------
        None

        """
        super()._log_infos(traj_infos)
        if traj_infos is None:
            traj_infos = self._traj_infos
        y_pred = None
        y_true = None
        if traj_infos is not None:
            y_pred = [info.get("_metric_y_pred", None) for info in traj_infos]
            y_true = [info.get("_metric_y_true", None) for info in traj_infos]
            y_pred_dist = [info.get("_dist_info", None) for info in traj_infos]
            y_pred_dist = [
                a.numpy() if isinstance(a, torch.Tensor) else a for a in y_pred_dist
            ]
            if (self.eval_metrics is not None) and len(y_pred) > 0:
                for metric in self.eval_metrics:
                    if metric is None:
                        continue
                    if metric.lower().startswith("top-"):
                        pred_info = y_pred_dist
                    else:
                        pred_info = y_pred
                    if not ((pred_info[0] is None) or (y_true[0] is None)):
                        result = self.metric_factory.evaluate(metric, y_true, pred_info)
                        logger.record_tabular(metric, result)


    def batch_evaluation(self):
        if hasattr(self.agent, "give_V_min_max"):
            if not self._setModelVMinMax and hasattr(self.agent.model, "set_V_min_max"):
                self.agent.model.set_V_min_max(self.agent.V_min, self.agent.V_max)
                self._setModelVMinMax = True
        self.agent.model.eval()
        
        if not hasattr(self, 'batchEvalkwargs'):
            self.batchEvalkwargs = None

        if self.batchEvalkwargs is None:
            self.batchEvalkwargs = dict()
            if getattr(self.algo, "reward_shaping_flag", False) and getattr(self.algo, "reward_shaping_coef", 1.0) != 0.0:
                self.batchEvalkwargs['explorationTemporalWeight'] = numpy_sigmoid_scheduler(
                    np.arange(self.batch_env.max_turns + 1), 
                    getattr(self.algo, "reward_shaping_kwargs", {}).get('js_alpha', 5),
                    self.batch_env.max_turns,
                    0,
                    getattr(self.algo, "reward_shaping_kwargs", {}).get('min_map_val', -10),
                    getattr(self.algo, "reward_shaping_kwargs", {}).get('max_map_val', 10),
                    is_decreasing=True,
                )
                self.batchEvalkwargs['weightExploration'] = getattr(self.algo, "reward_shaping_kwargs", {}).get('js_weight')
                self.batchEvalkwargs['min_exploration_reward'] = getattr(self.algo, "reward_shaping_kwargs", {}).get('bounds_dict', {}).get('js_min')
                self.batchEvalkwargs['max_exploration_reward'] = getattr(self.algo, "reward_shaping_kwargs", {}).get('bounds_dict', {}).get('js_max')

                self.batchEvalkwargs['confirmationTemporalWeight'] = numpy_sigmoid_scheduler(
                    np.arange(self.batch_env.max_turns + 1), 
                    getattr(self.algo, "reward_shaping_kwargs", {}).get('ce_alpha', 5),
                    self.batch_env.max_turns,
                    0,
                    getattr(self.algo, "reward_shaping_kwargs", {}).get('min_map_val', -10),
                    getattr(self.algo, "reward_shaping_kwargs", {}).get('max_map_val', 10),
                )
                self.batchEvalkwargs['weightConfirmation'] = getattr(self.algo, "reward_shaping_kwargs", {}).get('ce_weight')
                self.batchEvalkwargs['min_confirmation_reward'] = getattr(self.algo, "reward_shaping_kwargs", {}).get('bounds_dict', {}).get('ce_min')
                self.batchEvalkwargs['max_confirmation_reward'] = getattr(self.algo, "reward_shaping_kwargs", {}).get('bounds_dict', {}).get('ce_max')

                self.batchEvalkwargs['weightSeverity'] = getattr(self.algo, "reward_shaping_kwargs", {}).get('sev_out_weight')
                self.batchEvalkwargs['min_severity_reward'] = getattr(self.algo, "reward_shaping_kwargs", {}).get('bounds_dict', {}).get('sev_out_min')
                self.batchEvalkwargs['max_severity_reward'] = getattr(self.algo, "reward_shaping_kwargs", {}).get('bounds_dict', {}).get('sev_out_max')

            if getattr(self.algo, "clf_reward_flag", False) and getattr(self.algo, "clf_reward_coef", 1.0) != 0.0:
                self.batchEvalkwargs['weightClassification'] = getattr(self.algo, "clf_reward_coef", 1.0)
                self.batchEvalkwargs['min_classification_reward'] = None
                self.batchEvalkwargs['max_classification_reward'] = None
                self.batchEvalkwargs['weightSevIn'] = getattr(self.algo, "clf_reward_kwargs", {}).get('sev_in_weight')

            self.batchEvalkwargs['discount'] = getattr(self.algo, "discount", 1)

        result = batch_evaluate(
            self.batch_env,
            self.agent,
            self.batch_env.max_turns,
            seed=None,
            compute_metrics_flag=True,
            batch_size=self.eval_batch_size,
            deterministic=True,
            output_fp=None,
            **self.batchEvalkwargs 
        )
        self.agent.model.train()
        for k in result.keys():
            if isinstance(result[k], (float, int)):
                k1 = k.replace("@", "_")
                logger.record_tabular("BatchEVal_" + k1, result[k])
        perf = None
        for k in self.custom_batch_metrics:
            aPerf = 0.0
            for q in self.custom_batch_metrics.get(k, {}):
                aPerf += result.get(q, 0) * self.custom_batch_metrics.get(k, {}).get(q, 0.0)
            logger.record_tabular("BatchEValMetric_" + k, aPerf)
            if self.perf_metric == k:
                perf = aPerf
        return perf

    def get_eval_performance(self, *args, **kwargs):
        """Method for computing the current eval performance.

        Parameters
        ----------
        args: list
            list of arguments.
        kwargs: dict
            dict of  arguments.

        Returns
        -------
        performance: float, None
            the computed performance.

        """
        if self.batch_env is not None:
            performance = self.batch_evaluation()
            if performance is not None:
                return performance
        eval_traj_infos = None
        if self._eval:
            if isinstance(self, MinibatchRlBase):
                if len(args) > 0:
                    eval_traj_infos = args[0]
                else:
                    eval_traj_infos = kwargs.get("eval_traj_infos", None)
        traj_infos = (
            getattr(self, "_traj_infos", None)
            if eval_traj_infos is None
            else eval_traj_infos
        )
        if traj_infos:
            key_perf = (
                "Return"
                if (self.perf_metric is None) or (self.perf_metric == "Reward")
                else self.perf_metric
            )
            if key_perf in traj_infos[0]:
                performance = sum([info[key_perf] for info in traj_infos]) / max(
                    1, len(traj_infos)
                )
            else:
                if key_perf.lower().startswith("top-"):
                    y_pred = [info.get("_dist_info", None) for info in traj_infos]
                    y_pred = [
                        a.numpy() if isinstance(a, torch.Tensor) else a for a in y_pred
                    ]
                else:
                    y_pred = [info.get("_metric_y_pred", None) for info in traj_infos]
                y_true = [info.get("_metric_y_true", None) for info in traj_infos]
                if not ((y_pred[0] is None) or (y_true[0] is None)):
                    performance = self.metric_factory.evaluate(key_perf, y_true, y_pred)
                else:
                    performance = None
        else:
            performance = None

        return performance

    def save_eval_performance(self, performance, itr, sampler_itr=None):
        """Save the performance metric obtained at the current iteration.

        Parameters
        ----------
        performance: float
            the corresponding the performance metric.
        itr: int
            current iteration number
        sampler_itr: int, None
            current sampler iteration number.

        Returns
        -------
        None

        """
        # min_itr_learn is not defined in AsyncRlBase
        min_itr_learn = getattr(self, "min_itr_learn", 0)
        if itr >= min_itr_learn - 1:
            if performance is not None:
                if (self.best_eval_performance is None) or (
                    performance > self.best_eval_performance
                ):
                    self.best_eval_performance = performance
                    if sampler_itr is None:
                        params = self.get_itr_snapshot(itr)
                    else:
                        params = self.get_itr_snapshot(itr, sampler_itr)
                    file_name = osp.join(logger.get_snapshot_dir(), BEST_MODEL_NAME)
                    torch.save(params, file_name)
                    self.remaining_patience = self.max_patience
                else:
                    if self.max_patience is not None:
                        self.remaining_patience -= 1

        if performance is not None:
            self.last_perf_windows.append(performance)

    def early_stop(self):
        """Method to perform early stopping.
        """
        if (self.max_patience is not None) and (self.remaining_patience < 0):
            self.shutdown()
            raise EarlyStoppingError()

    def get_final_performance(self):
        """Method for computing the final performance of the training.

        This is computed as the mean of the last `perf_window_size`
        evaluation performances.

        Parameters
        ----------

        Returns
        -------
        performance: float, None
            the computed performance.

        """
        return sum(self.last_perf_windows) / max(1, len(self.last_perf_windows))

    def add_resume_info_to_snapshot(self, snapshot):
        """Method to add resume infos in the snapshot (checkpointing).

        Parameters
        ----------
        snapshot: dict
            current snapshot.

        Returns
        -------
        result: dict
            the updated snapshot.

        """
        resume_info = {}
        resume_info["best_eval_performance"] = self.best_eval_performance
        resume_info["itr"] = snapshot["itr"]
        if "sampler_itr" in snapshot:
            resume_info["sampler_itr"] = snapshot["sampler_itr"]
        resume_info["cum_updates"] = self.algo.update_counter
        if hasattr(self, "_cum_completed_trajs"):
            resume_info["cum_completed_trajs"] = self._cum_completed_trajs

        snapshot["resume_info"] = resume_info

        return snapshot

    def _pop_and_push_logger_prefix(self, itr, new_itr):
        """Modify the logger's logging prefix to account of resumed training experiment.

        Parameters
        ----------
        itr: int
            current iteration number.
        new_itr: int
            iteration number the logger need to be reset with.

        Returns
        -------
        None

        """
        if len(logger._prefixes) > 0 and logger._prefixes[-1].endswith(f"#{itr} "):
            old_str = logger._prefixes[-1]
            new_str = old_str.replace(f"#{itr} ", f"#{new_itr} ")
            logger.pop_prefix()
            logger.push_prefix(new_str)

    def activate_resume_logging(self, itr):
        """Modify the logging  to account for resumed training experiment.

        Parameters
        ----------
        itr: int
            current iteration number.

        Returns
        -------
        None

        """
        self._pop_and_push_logger_prefix(itr, itr + self.init_itr)
        logger.set_iteration(itr + self.init_itr)
        self.n_itr += self.init_itr
        self.algo.update_counter += self.init_cum_updates

    def deactivate_resume_logging(self, itr):
        """Reset the logging.

        Parameters
        ----------
        itr: int
            current iteration number.

        Returns
        -------
        None

        """
        self.algo.update_counter -= self.init_cum_updates
        self.n_itr -= self.init_itr
        logger.set_iteration(itr)
        self._pop_and_push_logger_prefix(itr + self.init_itr, itr)


class SyncRunnerMixin:
    """Extend Sync runner ability for properly logging in case of resuming experiments.
    """

    def get_itr_snapshot(self, itr):
        """Method for getting the current training snapshot (checkpointing).

        Parameters
        ----------
        itr: int
            current iteration number.

        Returns
        -------
        result: dict
            dictionary containg the snapshot.

        """
        result = super().get_itr_snapshot(itr)
        result = self.add_resume_info_to_snapshot(result)
        return result

    def log_diagnostics(self, itr, *args, **kwargs):
        """Log the diagnostic for the current iteration.

        Parameters
        ----------
        itr: int
            current iteration number.
        args: list
            list of arguments.
        kwargs: dict
            dict of  arguments.

        Returns
        -------
        None

        """
        self.activate_resume_logging(itr)

        perf_metric = self.get_eval_performance(*args, **kwargs)
        self.save_eval_performance(perf_metric, itr + self.init_itr)

        # call the super method
        super().log_diagnostics(itr + self.init_itr, *args, **kwargs)

        self.deactivate_resume_logging(itr)

        # do eventually the early stopping
        self.early_stop()


class AsyncRunnerMixin:
    """Extend Async runner ability for properly logging in case of resuming experiments.
    """

    def get_itr_snapshot(self, itr, sampler_itr):
        """Method for getting the current training snapshot (checkpointing).

        Parameters
        ----------
        itr: int
            current iteration number.
        sampler_itr: int
            current sampler iteration number.

        Returns
        -------
        result: dict
            dictionary containg the snapshot.

        """
        result = super().get_itr_snapshot(itr, sampler_itr)
        result = self.add_resume_info_to_snapshot(result)
        return result

    def log_diagnostics(self, itr, sampler_itr, *args, **kwargs):
        """Log the diagnostic for the current iteration.

        Parameters
        ----------
        itr: int
            current iteration number.
        sampler_itr: int
            current sampler iteration number.
        args: list
            list of arguments.
        kwargs: dict
            dict of  arguments.

        Returns
        -------
        None

        """
        self.activate_resume_logging(itr)

        perf_metric = self.get_eval_performance(*args, **kwargs)
        self.save_eval_performance(
            perf_metric, itr + self.init_itr, sampler_itr + self.init_sampler_itr
        )

        # call the super method
        super().log_diagnostics(
            itr + self.init_itr, sampler_itr + self.init_sampler_itr, *args, **kwargs
        )

        self.deactivate_resume_logging(itr)

        # do eventually the early stopping
        self.early_stop()


class PerformanceMinibatchRl(PerformanceMixin, SyncRunnerMixin, MinibatchRl):
    """Extension for the `MinibatchRl` runner class.

    This extension includes the additionnal functionalities inplemented by the
    `PerformanceMixin` and `SyncRunnerMixin` Mixin classes.
    """

    def __init__(
        self,
        *args,
        resume_info={},
        metrics=None,
        perf_window_size=1,
        patience=None,
        topk=1,
        eval_coeffs=None,
        perf_metric="Reward",
        traj_auxiliary_reward_flag=False,
        batch_env=None,
        eval_batch_size=None,
        **kwargs,
    ):
        """Instantiates an object.

        Parameters
        ----------
        resume_info: dict
            dictionary containing resuming infos. Default: dict().
        metrics: list, None
            list of metrics to monitor during training. Default: None
        perf_window_size: int
            the number of last evaluation performace to consider
            Default: 1
        patience: int
            the number of iterations we may tolerate performance degradiation.
            Useful for early stopping. if None, ealy_stopping is disable.
            Default: None
        topk: int
            the number of top predicted pathologies to be considered during evaluation.
            Default: 1
        eval_coeffs: list
            the coefficients to be used to weight each component of the aggregated
            evaluation metrics. If None, the components are equally weighted.
            These components are defined in the following order:
            - Differential Diagnosis score
            - Symptom Discovery score
            - Risk Factor Discovery score
            - Negative Response Pertinence score
            Default: None
        perf_metric: str
            the metric used to monitor the evaluation. if None, use the reward.
            Default: Reward
        traj_auxiliary_reward_flag: bool
            flag indicating whether or not to log auxiliary rewards (if computed)
            in trajectory infos.
            Default: False
        batch_env: env
            batch environment. Default: None
        eval_batch_size: int
            batch size for evaluation. Default: None

        """
        super().__init__(*args, **kwargs)
        self.set_resume_params(
            resume_info,
            metrics,
            perf_window_size,
            patience,
            topk,
            eval_coeffs,
            perf_metric,
        )
        self.set_flag_for_logging_trajectory_auxiliary_reward(
            traj_auxiliary_reward_flag
        )
        self.set_batch_eval_context(batch_env, eval_batch_size)


class PerformanceMinibatchRlEval(PerformanceMixin, SyncRunnerMixin, MinibatchRlEval):
    """Extension for the `MinibatchRlEval` runner class.

    This extension includes the additionnal functionalities inplemented by the
    `PerformanceMixin` and `SyncRunnerMixin` Mixin classes.
    """

    def __init__(
        self,
        *args,
        resume_info={},
        metrics=None,
        perf_window_size=1,
        patience=None,
        topk=1,
        eval_coeffs=None,
        perf_metric="Reward",
        traj_auxiliary_reward_flag=False,
        batch_env=None,
        eval_batch_size=None,
        **kwargs,
    ):
        """Instantiates an object.

        Parameters
        ----------
        resume_info: dict
            dictionary containing resuming infos. Default: dict()
        metrics: list, None
            list of metrics to monitor during training.
        perf_window_size: int
            the number of last evaluation performace to consider.
            Default: 1
        patience: int
            the number of iterations we may tolerate performance degradiation.
            Useful for early stopping. if None, ealy_stopping is disable.
            Default: None
        topk: int
            the number of top predicted pathologies to be considered during evaluation.
            Default: 1
        eval_coeffs: list
            the coefficients to be used to weight each component of the aggregated
            evaluation metrics. If None, the components are equally weighted.
            These components are defined in the following order:
            - Differential Diagnosis score
            - Symptom Discovery score
            - Risk Factor Discovery score
            - Negative Response Pertinence score
            Default: None
        perf_metric: str
            the metric used to monitor the evaluation. if None, use the reward.
            Default: Reward
        traj_auxiliary_reward_flag: bool
            flag indicating whether or not to log auxiliary rewards (if computed)
            in trajectory infos.
            Default: False
        batch_env: env
            batch environment. Default: None
        eval_batch_size: int
            batch size for evaluation. Default: None

        """
        super().__init__(*args, **kwargs)
        self.set_resume_params(
            resume_info,
            metrics,
            perf_window_size,
            patience,
            topk,
            eval_coeffs,
            perf_metric,
        )
        self.set_flag_for_logging_trajectory_auxiliary_reward(
            traj_auxiliary_reward_flag
        )
        self.set_batch_eval_context(batch_env, eval_batch_size)


class PerformanceSyncRl(PerformanceMixin, SyncRunnerMixin, SyncRl):
    """Extension for the `SyncRl` runner class.

    This extension includes the additionnal functionalities inplemented by the
    `PerformanceMixin` and `SyncRunnerMixin` Mixin classes.
    """

    def __init__(
        self,
        *args,
        resume_info={},
        metrics=None,
        perf_window_size=1,
        patience=None,
        topk=1,
        eval_coeffs=None,
        perf_metric="Reward",
        traj_auxiliary_reward_flag=False,
        batch_env=None,
        eval_batch_size=None,
        **kwargs,
    ):
        """Instantiates an object.

        Parameters
        ----------
        resume_info: dict
            dictionary containing resuming infos. Default: dict()
        perf_window_size: int
            the number of last evaluation performace to consider.
            Default: 1
        patience: int
            the number of iterations we may tolerate performance degradiation.
            Useful for early stopping. if None, ealy_stopping is disable.
            Default: None
        topk: int
            the number of top predicted pathologies to be considered during evaluation.
            Default: 1
        eval_coeffs: list
            the coefficients to be used to weight each component of the aggregated
            evaluation metrics. If None, the components are equally weighted.
            These components are defined in the following order:
            - Differential Diagnosis score
            - Symptom Discovery score
            - Risk Factor Discovery score
            - Negative Response Pertinence score
            Default: None
        perf_metric: str
            the metric used to monitor the evaluation. if None, use the reward.
            Default: Reward
        traj_auxiliary_reward_flag: bool
            flag indicating whether or not to log auxiliary rewards (if computed)
            in trajectory infos.
            Default: False
        batch_env: env
            batch environment. Default: None
        eval_batch_size: int
            batch size for evaluation. Default: None

        """
        super().__init__(*args, **kwargs)
        self.set_resume_params(
            resume_info,
            metrics,
            perf_window_size,
            patience,
            topk,
            eval_coeffs,
            perf_metric,
        )
        self.set_flag_for_logging_trajectory_auxiliary_reward(
            traj_auxiliary_reward_flag
        )
        self.set_batch_eval_context(batch_env, eval_batch_size)


class PerformanceSyncRlEval(PerformanceMixin, SyncRunnerMixin, SyncRlEval):
    """Extension for the `SyncRlEval` runner class.

    This extension includes the additionnal functionalities inplemented by the
    `PerformanceMixin` and `SyncRunnerMixin` Mixin classes.
    """

    def __init__(
        self,
        *args,
        resume_info={},
        metrics=None,
        perf_window_size=1,
        patience=None,
        topk=1,
        eval_coeffs=None,
        perf_metric="Reward",
        traj_auxiliary_reward_flag=False,
        batch_env=None,
        eval_batch_size=None,
        **kwargs,
    ):
        """Instantiates an object.

        Parameters
        ----------
        resume_info: dict
            dictionary containing resuming infos. Default: dict().
        metrics: list, None
            list of metrics to monitor during training. Default: None
        perf_window_size: int
            the number of last evaluation performace to consider.
            Default: 1
        patience: int
            the number of iterations we may tolerate performance degradiation.
            Useful for early stopping. if None, ealy_stopping is disable.
            Default: None
        topk: int
            the number of top predicted pathologies to be considered during evaluation.
            Default: 1
        eval_coeffs: list
            the coefficients to be used to weight each component of the aggregated
            evaluation metrics. If None, the components are equally weighted.
            These components are defined in the following order:
            - Differential Diagnosis score
            - Symptom Discovery score
            - Risk Factor Discovery score
            - Negative Response Pertinence score
            Default: None
        perf_metric: str
            the metric used to monitor the evaluation. if None, use the reward.
            Default: Reward
        traj_auxiliary_reward_flag: bool
            flag indicating whether or not to log auxiliary rewards (if computed)
            in trajectory infos.
            Default: False
        batch_env: env
            batch environment. Default: None
        eval_batch_size: int
            batch size for evaluation. Default: None

        """
        super().__init__(*args, **kwargs)
        self.set_resume_params(
            resume_info,
            metrics,
            perf_window_size,
            patience,
            topk,
            eval_coeffs,
            perf_metric,
        )
        self.set_flag_for_logging_trajectory_auxiliary_reward(
            traj_auxiliary_reward_flag
        )
        self.set_batch_eval_context(batch_env, eval_batch_size)


class PerformanceAsyncRl(PerformanceMixin, AsyncRunnerMixin, AsyncRl):
    """Extension for the `AsyncRl` runner class.

    This extension includes the additionnal functionalities inplemented by the
    `PerformanceMixin` and `AsyncRunnerMixin` Mixin classes.
    """

    def __init__(
        self,
        *args,
        resume_info={},
        metrics=None,
        perf_window_size=1,
        patience=None,
        topk=1,
        eval_coeffs=None,
        perf_metric="Reward",
        traj_auxiliary_reward_flag=False,
        batch_env=None,
        eval_batch_size=None,
        **kwargs,
    ):
        """Instantiates an object.

        Parameters
        ----------
        resume_info: dict
            dictionary containing resuming infos. default: dict().
        metrics: list, None
            list of metrics to monitor during training. Default: None
        perf_window_size: int
            the number of last evaluation performace to consider.
            Default: 1
        patience: int
            the number of iterations we may tolerate performance degradiation.
            Useful for early stopping. if None, ealy_stopping is disable.
            Default: None
        topk: int
            the number of top predicted pathologies to be considered during evaluation.
            Default: 1
        eval_coeffs: list
            the coefficients to be used to weight each component of the aggregated
            evaluation metrics. If None, the components are equally weighted.
            These components are defined in the following order:
            - Differential Diagnosis score
            - Symptom Discovery score
            - Risk Factor Discovery score
            - Negative Response Pertinence score
            Default: None
        perf_metric: str
            the metric used to monitor the evaluation. if None, use the reward.
            Default: Reward
        traj_auxiliary_reward_flag: bool
            flag indicating whether or not to log auxiliary rewards (if computed)
            in trajectory infos.
            Default: False
        batch_env: env
            batch environment. Default: None
        eval_batch_size: int
            batch size for evaluation. Default: None

        """
        super().__init__(*args, **kwargs)
        self.set_resume_params(
            resume_info,
            metrics,
            perf_window_size,
            patience,
            topk,
            eval_coeffs,
            perf_metric,
        )
        self.set_flag_for_logging_trajectory_auxiliary_reward(
            traj_auxiliary_reward_flag
        )
        self.set_batch_eval_context(batch_env, eval_batch_size)


class PerformanceAsyncRlEval(PerformanceMixin, AsyncRunnerMixin, AsyncRlEval):
    """Extension for the `AsyncRlEval` runner class.

    This extension includes the additionnal functionalities inplemented by the
    `PerformanceMixin` and `AsyncRunnerMixin` Mixin classes.
    """

    def __init__(
        self,
        *args,
        resume_info={},
        metrics=None,
        perf_window_size=1,
        patience=None,
        topk=1,
        eval_coeffs=None,
        perf_metric="Reward",
        traj_auxiliary_reward_flag=False,
        batch_env=None,
        eval_batch_size=None,
        **kwargs,
    ):
        """Instantiates an object.

        Parameters
        ----------
        resume_info: dict
            dictionary containing resuming infos. default: dict()
        metrics: list, None
            list of metrics to monitor during training. Default: None
        perf_window_size: int
            the number of last evaluation performace to consider.
            Default: 1
        patience: int
            the number of iterations we may tolerate performance degradiation.
            Useful for early stopping. if None, ealy_stopping is disable.
            Default: None
        topk: int
            the number of top predicted pathologies to be considered during evaluation.
            Default: 1
        eval_coeffs: list
            the coefficients to be used to weight each component of the aggregated
            evaluation metrics. If None, the components are equally weighted.
            These components are defined in the following order:
            - Differential Diagnosis score
            - Symptom Discovery score
            - Risk Factor Discovery score
            - Negative Response Pertinence score
            Default: None
        perf_metric: str
            the metric used to monitor the evaluation. if None, use the reward.
            Default: Reward
        traj_auxiliary_reward_flag: bool
            flag indicating whether or not to log auxiliary rewards (if computed)
            in trajectory infos.
            Default: False
        batch_env: env
            batch environment. Default: None
        eval_batch_size: int
            batch size for evaluation. Default: None

        """
        super().__init__(*args, **kwargs)
        self.set_resume_params(
            resume_info,
            metrics,
            perf_window_size,
            patience,
            topk,
            eval_coeffs,
            perf_metric,
        )
        self.set_flag_for_logging_trajectory_auxiliary_reward(
            traj_auxiliary_reward_flag
        )
        self.set_batch_eval_context(batch_env, eval_batch_size)


class RunnerFactory:
    """A factory for instanciating runner objects.

    The predefined runner classes are:
        - MinibatchRl
        - MinibatchRlEval
        - SyncRl
        - SyncRlEval
        - AsyncRl
        - AsyncRlEval

    Please, refer to https://rlpyt.readthedocs.io/en/latest/pages/runner.html
    for mor details.
    """

    def __init__(self):
        """Instantiates an object.
        """
        self._builders = {
            "minibatchrl": PerformanceMinibatchRl,
            "minibatchrleval": PerformanceMinibatchRlEval,
            "syncrl": PerformanceSyncRl,
            "syncrleval": PerformanceSyncRlEval,
            "asyncrl": PerformanceAsyncRl,
            "asyncrleval": PerformanceAsyncRlEval,
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

    def _get_affinity_for_minibacth_runner(
        self, sampler, cuda_idx, n_workers, cpu_list
    ):
        """Method for determining the resource affinity for minibacth ruuner.

        Parameters
        ----------
        sampler: object
            sampler object to be used during the experiment.
        cuda_idx: int
            index of the gpu to be used (Single-GPU optimization).
        n_workers: int
            number of cpu workers required in the experiments.
        cpu_list: list of int, None
            list of cpus to be used by the experiment.

        Return
        ------
        affinity: dict
            the resource affinity to be used for the experiment.

        """
        if not is_async_sampler(sampler):
            affinity = dict(cuda_idx=cuda_idx)
            if n_workers > 0:
                if (cpu_list is None) or (len(cpu_list) == 0):
                    p = psutil.Process()
                    cpu_affin = p.cpu_affinity()
                    affinity["workers_cpus"] = (
                        cpu_affin
                        if n_workers > len(cpu_affin)
                        else cpu_affin[0:n_workers]
                    )
                else:
                    if n_workers > len(cpu_list):
                        raise ValueError(
                            f"The number of specified workers {n_workers} "
                            f"must be less or equal than the number of "
                            f"specified cpus: {len(cpu_list)} - {cpu_list}."
                        )
                    else:
                        affinity["workers_cpus"] = cpu_list[0:n_workers]
            if is_aternating_sampler(sampler):
                if not (n_workers > 0):
                    raise ValueError(
                        f"for alternating sampler, the number expected of workers"
                        f"must be grater than zero: {n_workers} provided."
                    )
                affinity["alternating"] = True
                # (Double list)
                affinity["workers_cpus"] += affinity["workers_cpus"]
            affinity["set_affinity"] = True
            return affinity
        else:
            raise ValueError(
                "The provided runner (not async) is not compliant with "
                "the provided sampler (async)."
            )

    def _get_affinity_params(self, is_runner_async, sampler, n_workers, n_gpus):
        """Computes the parameters required to determine the affinity of the experiment.

        Parameters
        ----------
        is_runner_async: bool
            boolean indicating is the runner is asynchronour or not.
        sampler: object
            sampler object to be used during the experiment.
        n_workers: int
            number of cpu workers required in the experiments.
        n_gpus: int
            number of gpu to be used in the experiment (Multi-GPU optimization).

        Return
        ------
        affinity_params: dict
            the parameter to be used to compute the affinity
            needed for the experiment.

        """
        affininty_params = dict(run_slot=0)
        n_cpu_core = psutil.cpu_count(logical=False)
        n_gpu = torch.cuda.device_count()

        if n_cpu_core is None:
            n_cpu_core = n_workers
        if n_gpu > 0:
            if not is_runner_async and not (n_gpus > 1):
                raise ValueError(
                    f"A multi-GPU runner is requested but the number of"
                    f" specified gpu per run is less or equal than 1:"
                    f" {n_gpus}"
                )
            if is_runner_async and (n_gpus == 1):
                affininty_params["optim_sample_share_gpu"] = True
            if n_gpus > n_gpu:
                raise ValueError(
                    f"A multi-GPU runner is requested but the number of"
                    f" specified gpu per run ({n_gpus}) is greater than "
                    f"the number of available gpu ({n_gpu})"
                )
            gpu_per_run = n_gpus
            cpu_per_run = 1
        else:
            if not is_runner_async and not (n_workers > 1):
                raise ValueError(
                    f"A multi-CPU runner is requested but the number of"
                    f" specified workers per run is less or equal than 1: "
                    f"{n_workers}"
                )
            if n_workers > n_cpu_core:
                raise ValueError(
                    f"A multi-CPU runner is requested but the number of"
                    f" specified workers per run ({n_workers}) is greater than "
                    f"the number of available cpu ({n_cpu_core})"
                )
            cpu_per_run = n_workers
            gpu_per_run = 1

        if is_aternating_sampler(sampler):
            affininty_params["alternating"] = True

        if is_runner_async:
            affininty_params["async_sample"] = True
            affininty_params["sample_gpu_per_run"] = gpu_per_run
            affininty_params["cpu_reserved"] = 1

        affininty_params["n_cpu_core"] = n_cpu_core
        affininty_params["n_gpu"] = n_gpu
        affininty_params["gpu_per_run"] = gpu_per_run
        affininty_params["cpu_per_run"] = cpu_per_run
        affininty_params["set_affinity"] = True

        return affininty_params

    def get_affinity(self, key, sampler, cuda_idx, n_workers, n_gpus, cpu_list, thread):
        """Method for determining the resource affinity for the experiment.

        Parameters
        ----------
        key: str
            registration key of the runner to be used in the experiment.
        sampler: object
            sampler object to be used during the experiment.
        cuda_idx: int
            index of the gpu to be used (Single-GPU optimization).
        n_workers: int
            number of cpu workers required in the experiments.
        n_gpus: int
            number of gpu to be used in the experiment (Multi-GPU optimization).
        cpu_list: list of int, None
            list of cpus to be used by the experiment.
        thread: int
            number of threads to be used by pytorch (see `torch.set_num_threads()`).
            Set it to None for automatic detection.

        Return
        ------
        affinity: dict
            the resource affinity to be used for the experiment.

        """
        builder = self._builders.get(key.lower())
        if not builder:
            raise ValueError(key + " is not a valid runner key")

        affinity = None

        if issubclass(builder, MinibatchRlBase):
            affinity = self._get_affinity_for_minibacth_runner(
                sampler, cuda_idx, n_workers, cpu_list
            )
        else:
            is_runner_async = issubclass(builder, AsyncRlBase)
            is_sampler_async = is_async_sampler(sampler)

            async_combination = is_runner_async and is_sampler_async
            sync_combination = not is_runner_async and not is_sampler_async
            is_valid_combination = async_combination or sync_combination

            if is_valid_combination:
                affininty_params = self._get_affinity_params(
                    is_runner_async, sampler, n_workers, n_gpus
                )
                logger.log(f"aff param: {affininty_params}")
                affinity = make_affinity(**affininty_params)
            else:
                raise ValueError(
                    f'The provided runner ({"a" if is_runner_async else "" + "sync" }) '
                    f"is not compliant with the provided "
                    f'sampler ({"a" if is_sampler_async else "" + "sync" }).'
                )
        if (cpu_list is not None) and (len(cpu_list) > 0):
            affinity["master_cpus"] = cpu_list
        if (thread is not None) and (thread > 0):
            affinity["master_torch_threads"] = thread
        logger.log(f"affinity: {affinity}")
        return affinity

    def get_runner_class(self, key):
        """Get the runner class based on the provided key.

        Parameters
        ----------
        key: str
            key indication of which runner class to retrieve.

        Return
        ------
        cls: class
            the runner class associated with the provided key.

        """
        builder = self._builders.get(key.lower())
        if not builder:
            raise ValueError(key + " is not a valid runner key")
        return builder

    def create(
        self,
        key,
        *args,
        resume_info={},
        metrics=None,
        perf_window_size=1,
        patience=None,
        topk=1,
        eval_coeffs=None,
        perf_metric="Reward",
        traj_auxiliary_reward_flag=False,
        batch_env=None,
        eval_batch_size=None,
        **kwargs,
    ):
        """Get a runner instance based on the provided key.

        Parameters
        ----------
        key: str
            key indication of which runner instance to create.
        args: list
            list of arguments.
        resume_info: dict
            dictionary containing resuming infos. Default: dict()
        metrics: list, None
            list of metrics to monitor during training. Default: None
        perf_window_size: int
            the number of last evaluation performace to consider.
            Default: 1
        patience: int
            the number of iterations we may tolerate performance degradiation.
            Useful for early stopping. if None, ealy_stopping is disable.
            Default: None
        topk: int
            the number of top predicted pathologies to be considered during evaluation.
            Default: 1
        eval_coeffs: list
            the coefficients to be used to weight each component of the aggregated
            evaluation metrics. If None, the components are equally weighted.
            These components are defined in the following order:
            - Differential Diagnosis score
            - Symptom Discovery score
            - Risk Factor Discovery score
            - Negative Response Pertinence score
            Default: None
        perf_metric: str
            the metric used to monitor the evaluation. if None, use the reward.
            Default: Reward
        traj_auxiliary_reward_flag: bool
            flag indicating whether or not to log auxiliary rewards (if computed)
            in trajectory infos.
            Default: False
        batch_env: env
            batch environment. Default: None
        eval_batch_size: int
            batch size for evaluation. Default: None
        kwargs: dict
            dict of  arguments.

        Return
        ------
        ruuner: object
            the instantiated runner based on the provided key.

        """
        builder = self._builders.get(key.lower())
        if not builder:
            raise ValueError(key + " is not a valid runner key")
        custom_batch_metrics = kwargs.pop('custom_metrics', {})
        runner = builder(
            *args,
            resume_info=resume_info,
            metrics=metrics,
            perf_window_size=perf_window_size,
            patience=patience,
            topk=topk,
            eval_coeffs=eval_coeffs,
            perf_metric=perf_metric,
            traj_auxiliary_reward_flag=traj_auxiliary_reward_flag,
            batch_env=batch_env,
            eval_batch_size=eval_batch_size,
            **kwargs,
        )
        runner.custom_batch_metrics = custom_batch_metrics
        return runner
