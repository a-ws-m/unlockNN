"""Uncertainty quantification metrics and evaluation utilities."""
from typing import Callable, Dict, List, Optional

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
from pymatgen.core.structure import Structure

from .megnet_utils import Targets
from .model import ProbGNN


def neg_log_likelihood(
    predictions: Targets,
    stddevs: Targets,
    true_vals: Targets,
) -> float:
    """Calculate the negative log likelihood of true values given predictions."""
    dist = tfp.distributions.Normal(loc=predictions, scale=stddevs)
    nll = -tf.math.reduce_sum(dist.log_prob(true_vals))
    return nll.numpy()


def sharpness(
    predictions: Optional[Targets],
    stddevs: Targets,
    true_vals: Optional[Targets],
) -> float:
    """Calculate the sharpness of predictions.

    Sharpness is the RMS of the predicted standard deviations.

    """
    return np.sqrt(np.mean(np.square(stddevs)))


def variation(
    predictions: Optional[Targets],
    stddevs: Targets,
    true_vals: Optional[Targets],
) -> float:
    """Calculate the coefficient of variation of predictions.

    Indicates dispersion of uncertainty estimates.

    """
    stddev_mean = np.mean(stddevs)
    coeff_var = np.sqrt(np.sum(np.square(stddevs - stddev_mean)) / (len(stddevs) - 1))
    coeff_var /= stddev_mean
    return coeff_var


class MeanErrorMetric:
    """Handler for mean error metrics.

    Args:
        mean_what_func: What type of mean we're calculating.
            For example, 'mean absolute error' -> ``np.abs``.
        take_root: Whether to take the square root of the mean.

    """

    def __init__(
        self,
        mean_what_func: Callable[[np.ndarray], np.ndarray],
        take_root: bool = False,
    ) -> None:
        """Initialize attribute."""
        self.mean_what_func = mean_what_func
        self.take_root = take_root

    def __call__(
        self,
        predictions: Targets,
        stddevs: Optional[Targets],
        true_vals: Targets,
    ) -> float:
        """Calculate the mean metric."""
        mean_error = np.mean(self.mean_what_func(true_vals - predictions))
        if self.take_root:
            return np.sqrt(mean_error)
        else:
            return mean_error


MAE = MeanErrorMetric(np.abs)
MSE = MeanErrorMetric(np.square)
RMSE = MeanErrorMetric(np.square, True)

AVAILABLE_METRICS: Dict[str, Callable[[Targets, Targets, Targets], float]] = {
    "nll": neg_log_likelihood,
    "sharpness": sharpness,
    "variation": variation,
    "mae": MAE,
    "mse": MSE,
    "rmse": RMSE,
}


def evaluate_uq_metrics(
    prob_model: ProbGNN,
    test_structs: List[Structure],
    test_targets: Targets,
    metrics: List[str] = AVAILABLE_METRICS.keys(),
) -> Dict[str, float]:
    """Evaluate probabilistic model metrics."""
    metrics_dict = {metric: AVAILABLE_METRICS[metric] for metric in metrics}
    predictions, stddevs = prob_model.predict(test_structs)
    return {
        metric: func(predictions, stddevs, test_targets)
        for metric, func in metrics_dict.items()
    }
