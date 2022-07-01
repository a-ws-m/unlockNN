"""Uncertainty quantification metrics and evaluation utilities."""
from typing import Callable, Dict, List, Optional

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from pymatgen.core.structure import Structure

from .megnet_utils import Targets
from .model import ProbNN, ModelInput


def neg_log_likelihood(
    predictions: Targets,
    stddevs: Targets,
    true_vals: Targets,
) -> float:
    r"""Calculate the negative log likelihood (NLL) of true values given predictions.

    NLL is given by

    .. math::

        \mathrm{NLL} = -\sum_i \log p_i(y_i),

    where :math:`y_i` is the :math:`i^\mathrm{th}` observed (true) value and
    :math:`p_i` is the probability density function for the
    :math:`i^\mathrm{th}` predicted Gaussian distribution:

    .. math::

        p_i \sim \mathcal{N} \left( \hat{y}_i, \sigma_i^2 \right),

    where :math:`\hat{y}_i` is the :math:`i^\mathrm{th}` predicted mean and
    :math:`\sigma_i` is the :math:`i^\mathrm{th}` predicted standard deviation.

    """
    dist = tfp.distributions.Normal(loc=predictions, scale=stddevs)
    nll = -tf.math.reduce_sum(dist.log_prob(true_vals))
    return nll.numpy()


def sharpness(
    predictions: Optional[Targets],
    stddevs: Targets,
    true_vals: Optional[Targets],
) -> float:
    """Calculate the sharpness of predictions.

    Sharpness is the root-mean-squared of the predicted standard deviations.

    """
    return np.sqrt(np.mean(np.square(stddevs)))


def variation(
    predictions: Optional[Targets],
    stddevs: Targets,
    true_vals: Optional[Targets],
) -> float:
    r"""Calculate the coefficient of variation of predictions.

    Indicates dispersion of uncertainty estimates.

    Let :math:`\sigma` be predicted standard deviations, :math:`\bar{\sigma}` be
    the mean of the standard deviations and :math:`N` be the number of
    predictions. The coefficient of variation is given by:

    .. math:: C_v = \frac{1}{\bar{\sigma}}\sqrt{\frac{\sum_i^N{(\sigma_i - \bar{\sigma})^2}}{N - 1}}

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
        mean_error = np.mean(self.mean_what_func(np.array(true_vals) - np.array(predictions)))
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
"""Indicates the mapping between a metric name (a potential argument to
:func:`evaluate_uq_metrics`) and the function it calls."""


def evaluate_uq_metrics(
    prob_model: ProbNN,
    test_inputs: List[ModelInput],
    test_targets: Targets,
    metrics: List[str] = list(AVAILABLE_METRICS.keys()),
) -> Dict[str, float]:
    """Evaluate probabilistic model metrics.

    Args:
        prob_model: The probabilistic model to evaluate.
        test_inputs: The input structures or graphs.
        test_targets: The target values for the structures.
        metrics: A list of metrics to compute. Defaults
            to computing all of the currently implemented
            metrics.

    Currently implemented metrics are given in :const:`AVAILABLE_METRICS`.

    Returns:
        Dictionary of ``{metric_name: value}``.

    Example:
        Compute the metrics of the example ``MEGNetProbModel`` for
        predicting binary compounds' formation energies:

        >>> from unlocknn.download import load_data, load_pretrained
        >>> binary_model = load_pretrained("binary_e_form")
        >>> binary_data = load_data("binary_e_form")
        >>> metrics = evaluate_uq_metrics(
        ...     binary_model, binary_data["structure"], binary_data["formation_energy_per_atom"]
        ... )
        >>> for metric_name, value in metrics.items():
        ...     print(f"{metric_name} = {value:.3f}")
        nll = -8922.768
        sharpness = 0.032
        variation = 0.514
        mae = 0.027
        mse = 0.002
        rmse = 0.041


    """
    metrics_dict = {metric: AVAILABLE_METRICS[metric] for metric in metrics}
    predictions, stddevs = prob_model.predict(test_inputs)
    return {
        metric: func(predictions, stddevs, test_targets)
        for metric, func in metrics_dict.items()
    }


if __name__ == "__main__":  # pragma: no cover
    import doctest

    doctest.testmod()
