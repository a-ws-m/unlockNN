"""Metric analyzing tools."""
from pathlib import Path
from typing import Optional, Union, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from ..gp.gp_trainer import GPTrainer
from .visualisation import plot_calibration, plot_sharpness

tfd = tfp.distributions


class GPMetrics:
    """Handler for GP metric calculations.

    Many of the metrics herein are based upon implementations by `Train et al.`_,
    for metrics proposed by `Kuleshov et al.`_.

    Args:
        val_points (:obj:`tf.Tensor`): The validation indices.
        val_obs (:obj:`np.ndarray`): The validation observed true values.
        gp_trainer (:obj:`GPTrainer`): The :obj:`GPTrainer` instance to
            analyse.

    Attributes:
        val_points (:obj:`tf.Tensor`): The validation indices.
        val_obs (:obj:`np.ndarray`): The validation observed true values.
        gp_trainer (:obj:`GPTrainer`): The :obj:`GPTrainer` instance to
            analyse.
        gprm (:obj:`GaussianProcessRegressionModel`): A regression model from
            `gp_trainer` fit to the `val_points`.

    .. _Tran et al.:
        https://arxiv.org/abs/1912.10066
    .. _Kuleshov et al.:
        https://arxiv.org/abs/1807.00263

    """

    # Set of which properties need the mean and standard deviation to be updated
    REQUIRES_MEAN = {"mae", "calibration_err", "residuals", "pis"}
    REQUIRES_STDDEV = {"sharpness", "variation"}

    def __init__(
        self, val_points: tf.Tensor, val_obs: tf.Tensor, gp_trainer: GPTrainer
    ):
        """Initialize validation points and observations and the wrapped `GPTrainer`."""
        self.val_points = val_points
        self.val_obs = val_obs
        self.gp_trainer = gp_trainer
        self.gprm = gp_trainer.get_model(val_points)  # Instantiate GPRM
        self.update_mean()
        self.update_stddevs()

    def update_mean(self):
        """Update the GPRM mean predictions."""
        self.mean = self.gprm.mean().numpy()

    def update_stddevs(self):
        """Update the GPRM standard deviation predictions."""
        self.stddevs = self.gprm.stddev().numpy()

    @property
    def nll(self) -> float:
        """Calculate the negative log likelihood of observed true values.

        Returns:
            nll (float)

        """
        return -self.gprm.log_prob(self.val_obs).numpy()

    @property
    def mae(self) -> float:
        """Calculate the mean average error of predicted values.

        Returns:
            mean (float)

        """
        return tf.losses.mae(self.val_obs, self.mean).numpy()

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
