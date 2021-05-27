"""Metric analyzing tools."""
from pathlib import Path
from typing import Optional, Union, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.metrics import mean_absolute_error

from .visualisation import plot_calibration, plot_sharpness

tfd = tfp.distributions


def calculate_mean(vgp: tfp.layers.VariationalGaussianProcess):
    """Get the mean values from a distribution."""
    obs_index_points = vgp._index_points
    obs_index_points = tf.convert_to_tensor(
        obs_index_points, dtype=obs_index_points._dtype, name="observation_index_points"
    )
    return vgp.mean(index_points=obs_index_points)


def calculate_stddev(vgp: tfp.layers.VariationalGaussianProcess):
    """Get the mean values from a distribution."""
    obs_index_points = vgp._index_points
    obs_index_points = tf.convert_to_tensor(
        obs_index_points, dtype=obs_index_points._dtype, name="observation_index_points"
    )
    return vgp.stddev(index_points=obs_index_points)


class Sharpness(tf.keras.metrics.Metric):
    """Metric class for sharpness."""

    def __init__(self, name="sharpness", **kwargs):
        super().__init__(name=name, **kwargs)
        self.squared_std_devs = self.add_weight(
            name="squared_stddevs", initializer="zeros"
        )
        self.num_data_points = self.add_weight(
            name="sha_num_data_points", initializer="zeros"
        )

    def update_state(self, y_true, predicted_distribution, sample_weight=None):
        std_devs = calculate_stddev(predicted_distribution)
        self.squared_std_devs.assign_add(np.sum(np.square(std_devs)))
        self.num_data_points.assign_add(len(std_devs))

    def result(self):
        return np.sqrt(self.squared_std_devs / self.num_data_points)

    def reset_states(self):
        self.squared_std_devs.assign(0)
        self.num_data_points.assign(0)


class MetricAnalyser:
    """Handler for metric calculations.

    Many of the metrics herein are based upon implementations by `Tran et al.`_,
    for metrics proposed by `Kuleshov et al.`_.
    The values of :attr:`mean` and :attr:`stddevs` are not automatically updated when
    :attr:`dist` is updated, in order to save computing time.
    :meth:`update_mean` and :meth:`update_stddevs` must be called in order to update
    them when current values are needed.

    Args:
        val_points (:obj:`tf.Tensor`): The validation indices.
        val_obs (:obj:`np.ndarray`): The validation observed true values.
        dist (:obj:`Distribution`): The :obj:`Distribution` instance to
            analyse.
        calc_mean_on_init (bool): Whether to calculate the :attr:`dist`'s
            means on initialization.
        calc_stddev_on_init (bool): Whether to calculate the :attr:`dist`'s
            stddevs on initialization.

    Attributes:
        val_points (:obj:`tf.Tensor`): The validation indices.
        val_obs (:obj:`np.ndarray`): The validation observed true values.
        dist (:obj:`Distribution`): The :obj:`Distribution` instance to
            analyse.
        mean (:obj:`np.ndarray`): The means of the predicted distributions around
            `val_points`.
        stddevs (:obj:`np.ndarray`): The standard deviations of the predicted
            distributions around :attr:`val_points`.
        REQUIRES_MEAN (set of str): The metrics that require :attr:`mean` in order
            to calculate.
        REQUIRES_STDDEV (set of str): The metrics that require :attr:`stddev` in
            order to calculate.

    .. _Tran et al.:
        https://arxiv.org/abs/1912.10066
    .. _Kuleshov et al.:
        https://arxiv.org/abs/1807.00263

    """

    # Set of which properties need the mean and standard deviation to be updated
    REQUIRES_MEAN = {"mae", "calibration_err", "residuals", "pis"}
    REQUIRES_STDDEV = {"sharpness", "variation", "calibration_err"}

    def __init__(
        self,
        val_points: tf.Tensor,
        val_obs: tf.Tensor,
        dist: tfp.python.distributions.Distribution,
        calc_mean_on_init: bool = True,
        calc_stddev_on_init: bool = True,
    ):
        """Initialize attributes and mean + stddev predictions."""
        self.val_points = val_points
        self.val_obs = val_obs
        self.dist = dist

        if calc_mean_on_init:
            self.update_mean()
        if calc_stddev_on_init:
            self.update_stddevs()

    def update_mean(self):
        """Update the mean predictions."""
        self.mean: np.ndarray = self.dist.mean().numpy()

    def update_stddevs(self):
        """Update the standard deviation predictions."""
        self.stddevs: np.ndarray = self.dist.stddev().numpy()

    @property
    def nll(self) -> float:
        """Calculate the negative log likelihood of observed true values.

        Returns:
            nll (float)

        """
        return -self.dist.log_prob(self.val_obs).numpy()

    @property
    def mae(self) -> float:
        """Calculate the mean average error of predicted values.

        Returns:
            mean (float)

        """
        return mean_absolute_error(self.val_obs, self.mean)

    @property
    def mape(self) -> float:
        """Calculate the mean average percentage error of predicted values.

        Returns:
            mape (float)
        """
        return np.mean(np.abs(self.residuals / self.val_obs.numpy())) * 100

    @property
    def sharpness(self) -> float:
        """Calculate the root-mean-squared of predicted standard deviations.

        Returns:
            sharpness (float)

        """
        return np.sqrt(np.mean(np.square(self.stddevs)))

    @property
    def variation(self) -> float:
        """Calculate the coefficient of variation of the regression model.

        Indicates dispersion of uncertainty estimates.

        Returns:
            coeff_var (float)

        """
        stdev_mean = self.stddevs.mean()
        coeff_var = np.sqrt(np.sum(np.square(self.stddevs - stdev_mean)))
        coeff_var /= stdev_mean * (len(self.stddevs) - 1)
        return coeff_var

    @property
    def calibration_err(self) -> float:
        """Calculate the calibration error of the model.

        Calls :meth:`pis`, which is relatively slow.

        Returns:
            calibration_error (float)

        """
        predicted_pi, observed_pi = self.pis
        return np.sum(np.square(predicted_pi - observed_pi))

    @property
    def residuals(self) -> np.ndarray:
        """Calculate the residuals.

        Returns:
            residuals (:obj:`np.ndarray`): The difference between the means
                of the predicted distributions and the true values.

        """
        return self.mean - self.val_obs.numpy()

    def sharpness_plot(self, fname: Optional[Union[str, Path]] = None):
        """Plot the distribution of standard deviations and the sharpness.

        Args:
            fname (str or :obj:`Path`, optional): The name of the file to save to.
                If omitted, will show the plot after completion.

        """
        plot_sharpness(self.stddevs, self.sharpness, self.variation, fname)

    def calibration_plot(self, fname: Optional[Union[str, Path]] = None):
        """Plot the distribution of residuals relative to the expected distribution.

        Args:
            fname (str or :obj:`Path`, optional): The name of the file to save to.
                If omitted, will show the plot after completion.

        """
        predicted_pi, observed_pi = self.pis
        plot_calibration(predicted_pi, observed_pi, fname)

    @property
    def pis(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the percentile interval densities of a model.

        Based on the implementation by `Tran et al.`_. Initially proposed by `Kuleshov et al.`_.

        Args:
            residuals (:obj:`np.ndarray`): The normalised residuals of the model predictions.

        Returns:
            predicted_pi (:obj:`np.ndarray`): The percentiles used.
            observed_pi (:obj:`np.ndarray`): The density of residuals that fall within each of the
            `predicted_pi` percentiles.

        .. _Tran et al.:
            https://arxiv.org/abs/1912.10066
        .. _Kuleshov et al.:
            https://arxiv.org/abs/1807.00263

        """
        norm_resids = self.residuals / self.stddevs  # Normalise residuals

        norm = tfd.Normal(0, 1)  # Standard normal distribution

        predicted_pi = np.linspace(0, 1, 100)
        bounds = norm.quantile(
            predicted_pi
        ).numpy()  # Find the upper bounds for each percentile

        observed_pi = np.array(
            [np.count_nonzero(norm_resids <= bound) for bound in bounds]
        )  # The number of residuals that fall within each percentile
        observed_pi = (
            observed_pi / norm_resids.size
        )  # The fraction (density) of residuals that fall within each percentile

        return predicted_pi, observed_pi
