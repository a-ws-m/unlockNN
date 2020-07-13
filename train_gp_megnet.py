"""Utilities for training a GP fed from the MEGNet Concatenation layer for a pretrained model."""
from operator import itemgetter
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.python.util.deprecation as deprecation
import tensorflow_probability as tfp
from tqdm import tqdm

from .data_vis import plot_calibration, plot_sharpness

deprecation._PRINT_DEPRECATION_WARNINGS = False


tfd = tfp.distributions
tfk = tfp.math.psd_kernels


def convert_index_points(array: np.ndarray) -> tf.Tensor:
    """Reshape an array into a tensor appropriate for GP index points.

    Extends the amount of dimensions by `array.shape[1] - 1` and converts
    to a `Tensor` with `dtype=tf.float64`.

    Args:
        array (:obj:`np.ndarray`): The array to extend.

    Returns
        tensor (:obj:`tf.Tensor`): The converted Tensor.

    """
    shape = array.shape
    shape += (1,) * (shape[1] - 1)
    return tf.constant(array, dtype=tf.float64, shape=shape)


class GPTrainer(tf.Module):
    """Class for training hyperparameters for GP kernels."""

    def __init__(
        self,
        observation_index_points: tf.Tensor,
        observations: tf.Tensor,
        checkpoint_dir: Optional[str] = None,
    ):
        """Initialze attributes, kernel, optimizer and checkpoint manager."""
        self.observation_index_points = tf.Variable(
            observation_index_points,
            dtype=tf.float64,
            trainable=False,
            name="observation_index_points",
        )
        self.observations = tf.Variable(
            observations, dtype=tf.float64, trainable=False, name="observations",
        )

        self.amplitude = tf.Variable(1.0, dtype=tf.float64, name="amplitude")
        self.length_scale = tf.Variable(1.0, dtype=tf.float64, name="length_scale")

        # TODO: Customizable kernel
        self.kernel = tfk.MaternOneHalf(
            amplitude=self.amplitude,
            length_scale=self.length_scale,
            feature_ndims=self.observation_index_points.shape[1],
        )

        self.optimizer = tf.optimizers.Adam()

        self.training_steps = tf.Variable(
            0, dtype=tf.int32, trainable=False, name="training_steps"
        )

        self.val_nll = tf.Variable(
            np.inf, dtype=tf.float64, trainable=False, name="validation_nll"
        )
        self.val_mae = tf.Variable(
            np.inf, dtype=tf.float64, trainable=False, name="validation_mae"
        )
        self.val_sharpness = tf.Variable(
            np.inf, dtype=tf.float64, trainable=False, name="validation_sharpness"
        )
        self.val_coeff_var = tf.Variable(
            np.inf, dtype=tf.float64, trainable=False, name="validation_coeff_variance"
        )
        self.val_cal_err = tf.Variable(
            np.inf,
            dtype=tf.float64,
            trainable=False,
            name="validation_calibration_error",
        )

        if checkpoint_dir:
            self.ckpt = tf.train.Checkpoint(
                step=self.training_steps,
                amp=self.amplitude,
                ls=self.length_scale,
                val_nll=self.val_nll,
                val_mae=self.val_mae,
                val_sharpness=self.val_sharpness,
                val_coeff_var=self.val_coeff_var,
                val_cal_err=self.val_cal_err,
            )
            self.ckpt_manager = tf.train.CheckpointManager(
                self.ckpt,
                checkpoint_dir,
                max_to_keep=1,
                step_counter=self.training_steps,
            )

            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            if self.ckpt_manager.latest_checkpoint:
                print(f"Restored from {self.ckpt_manager.latest_checkpoint}")
            else:
                print("No checkpoints found.")

        else:
            self.ckpt = None
            self.ckpt_manager = None

        self.gp_prior = tfd.GaussianProcess(self.kernel, self.observation_index_points)

    @staticmethod
    def load_model(model_dir: str):
        """Load a `GPTrainer` model from a file."""
        return tf.saved_model.load(model_dir)

    def get_model(
        self, index_points: tf.Tensor
    ) -> tfp.python.distributions.GaussianProcessRegressionModel:
        """Get a regression model for a set of index points."""
        return tfd.GaussianProcessRegressionModel(
            kernel=self.kernel,
            index_points=index_points,
            observation_index_points=self.observation_index_points,
            observations=self.observations,
        )

    @tf.function(input_signature=[tf.TensorSpec(None, tf.float64)])
    def predict(self, points: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Predict values for given `points` and the standard deviation of the distribution."""
        gprm = self.get_model(points)
        return gprm.mean(), gprm.stddev()

    @staticmethod
    def calc_pis(residuals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
        norm = tfd.Normal(0, 1)

        predicted_pi = np.linspace(0, 1, 100)
        bounds = norm.quantile(predicted_pi).numpy()

        observed_pi = np.array(
            [np.count_nonzero(residuals <= bound) for bound in bounds]
        )
        observed_pi = observed_pi / residuals.size

        return predicted_pi, observed_pi

    def metrics(
        self, val_points: tf.Tensor, val_obs: tf.Tensor, make_plots: bool = False
    ) -> Tuple[float, float, float, float, float]:
        """Calculate model performance metrics.

        Returns:
            nll (float): Negative log likelihood.
            mae (float): Mean average error.
            sharpness (float): RMS of predicted standard deviations.
            coeff_var (float): Coefficient of variation. Indicates dispersion of
                uncertainty estimates.
            cal_error (float): Calibration error. Indicates how well the true
                frequency of points in each interval correspond to the predicted
                fraction of points in that interval (`Kuleshov et al.`_).

        .. _Kuleshov et al.:
            https://arxiv.org/abs/1807.00263

        """
        gprm = self.get_model(val_points)

        nll = -gprm.log_prob(val_obs).numpy()

        mae = tf.losses.mae(val_obs, gprm.mean().numpy()).numpy()

        stdevs = gprm.stddev().numpy()

        sharpness = np.sqrt(np.mean(np.square(stdevs)))

        stdev_mean = np.mean(stdevs)
        coeff_var = np.sqrt(np.sum(np.square(stdevs - stdev_mean)))
        coeff_var /= stdev_mean * (len(stdevs) - 1)

        resids = gprm.mean().numpy() - val_obs.numpy()
        resids /= stdevs  # Normalise residuals
        predicted_pi, observed_pi = self.calc_pis(resids)
        cal_error = np.sum(np.square(predicted_pi - observed_pi))

        if make_plots:
            plot_calibration(predicted_pi, observed_pi, "misc/calibration.pdf")
            plot_sharpness(stdevs, sharpness, coeff_var, "misc/sharpness.pdf")

        return nll, mae, sharpness, coeff_var, cal_error

    def train_model(
        self,
        val_points: tf.Tensor,
        val_obs: tf.Tensor,
        epochs: int = 1000,
        patience: Optional[int] = None,
        save_dir: Optional[Union[str, Path]] = None,
    ) -> Iterator[Tuple[float, float, float, float, float]]:
        """Optimize the parameters and measure validation metrics at each step."""
        best_nll: float = self.val_nll.numpy()
        steps_since_improvement: int = 1

        for i in tqdm(range(epochs), "Training epochs"):
            self.optimize_cycle()
            self.training_steps.assign_add(1)

            # * Determine and assign metrics
            metrics = self.metrics(val_points, val_obs)
            metric_order = (
                self.val_nll,
                self.val_mae,
                self.val_sharpness,
                self.val_coeff_var,
                self.val_cal_err,
            )
            for variable, metric in zip(metric_order, metrics):
                variable.assign(metric)

            yield metrics

            if self.val_nll < best_nll:
                best_nll = self.val_nll.numpy()
                steps_since_improvement = 1
                if self.ckpt_manager:
                    self.ckpt_manager.save(self.training_steps)
            else:
                steps_since_improvement += 1
                if patience and steps_since_improvement >= patience:
                    print(
                        "Patience exceeded: "
                        f"{steps_since_improvement} steps since NLL improvement."
                    )
                    break

        if save_dir:
            tf.saved_model.save(self, save_dir)

    @tf.function
    def optimize_cycle(self) -> tf.Tensor:
        """Perform one training step.

        Returns:
            loss (:obj:`Tensor`): A Tensor containing the negative log probability loss.

        """
        with tf.GradientTape() as tape:
            loss = -self.gp_prior.log_prob(self.observations)

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss
