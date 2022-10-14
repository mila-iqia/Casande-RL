from collections import OrderedDict

import numpy as np
from rlpyt.utils.logging import logger
from sklearn import metrics


class TopKAccuracy:
    """A Generic class for computing top-k accuracy.

    """

    def __init__(self, k):
        """Instantiates a class object.

        Parameters
        ----------
        k: int
            the number of top values to be considered.

        """
        if k <= 0:
            raise ValueError(f"Expected positive k, got {k}")
        self.k = k

    def __call__(self, y_true, y_pred, **kwargs):
        """A function for computing the top-k accuracy score.

        Parameters
        ----------
        y_true: vector
            ground truth classes.
        y_pred: vector
            predicted scores for all the classes.
        kwargs: dict
            additional parameters.

        Return
        ------
        result: float
            the top-k accuracy score.

        """
        topk = np.argsort(y_pred, axis=1)[:, -self.k :]
        y_true = np.array(y_true)
        result = (topk == y_true[..., None]).any(axis=1)
        result = result.astype(int)
        acc = sum(result) / max(result.shape[0], 1)
        return acc


def f1_score(y_true, y_pred, average="macro", **kwargs):
    """A function for computing the f1 score.

    Parameters
    ----------
    y_true: vector
        ground truth values.
    y_pred: vector
        predicted values.
    average: str
        determines the type of averaging performed on the data.
        Default: "macro"
    kwargs: dict
        additional parameters.

    Return
    ------
    result: float
        the computed f1 score.

    """
    return metrics.f1_score(y_true, y_pred, average=average, **kwargs)


def precision_score(y_true, y_pred, average="macro", **kwargs):
    """A function for computing the precision score.

    Parameters
    ----------
    y_true: vector
        ground truth values.
    y_pred: vector
        predicted values.
    average: str
        determines the type of averaging performed on the data.
        Default: "macro".
    kwargs: dict
        additional parameters.

    Return
    ------
    result: float
        the computed precision score.

    """
    return metrics.precision_score(y_true, y_pred, average=average, **kwargs)


def recall_score(y_true, y_pred, average="macro", **kwargs):
    """A function for computing the recall score.

    Parameters
    ----------
    y_true: vector
        ground truth values.
    y_pred: vector
        predicted values.
    average: str
        determines the type of averaging performed on the data.
        Default: "macro"
    kwargs: dict
        additional parameters.

    Return
    ------
    result: float
        the computed recall score.

    """
    return metrics.recall_score(y_true, y_pred, average=average, **kwargs)


class MetricFactory:
    """A factory for evaluating different metrics of interest.

    The predefined metric functions are:
        - accuracy
        - balanced_accuracy
        - f1
        - precision
        - recall
        - confusion_matrix
    """

    def __init__(self):
        self._builders = {
            "accuracy": metrics.accuracy_score,
            "balanced_accuracy": metrics.balanced_accuracy_score,
            "f1": f1_score,
            "precision": precision_score,
            "recall": recall_score,
            "confusion_matrix": metrics.confusion_matrix,
        }
        for k in range(1, 21):
            self._builders[f"top-{k}-accuracy"] = TopKAccuracy(k)

    def register_builder(self, key, builder, force_replace=False):
        """Register an instance within the factory.

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

    def evaluate(self, key, y_true, y_pred, **kwargs):
        """Evaluate a metric (defined by the provided key) given the provided data.

        Parameters
        ----------
        key: str
            key indication of which metric to use for the evaluation.
        y_true: list
            ground truth labels.
        y_pred: list
            predicted labels.

        Return
        ------
        result: float
            the computed metric score.

        """
        builder = self._builders.get(key.lower())
        if not builder:
            raise ValueError(key + " is not a valid metric key")
        return builder(y_true, y_pred, **kwargs)

    def misc_stats(self, values, prefix="", suffix=""):
        """Compute the stats of an array `values`.

        The stats include the mean, std, max, min, median of
        the provided array.

        Parameters
        ----------
        values: list, np.ndarray
            values for which we want to compute these stats.
        prefix: str
            prefix for indexation. Default: ""
        suffix: str
            suffix for indexation. Default: ""

        Return
        ------
        result: dict
            dictionary containing the computed stats.

        """
        result = OrderedDict()
        if len(values) > 0:
            result[prefix + "Avg" + suffix] = np.average(values)
            result[prefix + "Std" + suffix] = np.std(values)
            result[prefix + "Median" + suffix] = np.median(values)
            result[prefix + "Min" + suffix] = float(np.min(values))
            result[prefix + "Max" + suffix] = float(np.max(values))
            result[prefix + "Q25" + suffix] = float(np.percentile(values, 25))
            result[prefix + "Q75" + suffix] = float(np.percentile(values, 75))
        else:
            result[prefix + "Avg" + suffix] = np.nan
            result[prefix + "Std" + suffix] = np.nan
            result[prefix + "Median" + suffix] = np.nan
            result[prefix + "Min" + suffix] = np.nan
            result[prefix + "Max" + suffix] = np.nan
            result[prefix + "Q25" + suffix] = np.nan
            result[prefix + "Q75" + suffix] = np.nan

        return result

    def unique_counts(self, values):
        """Compute the unique count of element within the provided array `values`.

        Parameters
        ----------
        values: list, np.ndarray
            values for which we want to compute these stats.

        Return
        ------
        result: dict
            dictionary containing the counts of unique elements of the provided array.

        """
        if values is None or len(values) == 0:
            return {}
        if not hasattr(values[0], "__len__"):
            unique, counts = np.unique(values, return_counts=True)
            result = dict(zip(unique, counts))
            result = {int(k): int(result[k]) for k in result.keys()}
            return result

        num_elts = len(values)
        data = np.concatenate(values)
        data = data.astype(np.int)
        unique, counts = np.unique(data, return_counts=True)
        result = dict(zip(unique, counts))
        result = {int(k): int(result[k]) / num_elts for k in result.keys()}

        return result
