"""Utilities for training a GP fed from the MEGNet Concatenation layer for a pretrained model."""
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import tensorflow.python.util.deprecation as deprecation
import tensorflow_probability as tfp
from tqdm import tqdm

from ..datalib.metrics import MetricAnalyser

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
    """Class for training hyperparameters for GP kernels.

    Args:
        observation_index_points (:obj:`tf.Tensor`): The observed index points (_x_ values).
        observations (:obj:`tf.Tensor`): The observed samples (_y_ values).
        checkpoint_dir (str or :obj:`Path`, optional): The directory to check for
            checkpoints and to save checkpoints to.

    Attributes:
        observation_index_points (:obj:`tf.Tensor`): The observed index points (_x_ values).
        observations (:obj:`tf.Tensor`): The observed samples (_y_ values).
        checkpoint_dir (str or :obj:`Path`, optional): The directory to check for
            checkpoints and to save checkpoints to.
        amplitude (:obj:`tf.Tensor`): The amplitude of the kernel.
        length_scale (:obj:`tf.Tensor`): The length scale of the kernel.
        kernel (:obj:`tf.Tensor`): The kernel to use for the Gaussian process.
        optimizer (:obj:`Optimizer`): The optimizer to use for determining
            :attr:`amplitude` and :attr:`length_scale`.
        training_steps (:obj:tf.Tensor): The current number of training epochs executed.
        loss (:obj:`tf.Tensor`): The current loss on the training data
            (A negative log likelihood).
        metrics (dict): Contains metric names and values.
            Default to `np.nan` when uncalculated.
        ckpt (:obj:`Checkpoint`, optional): A tensorflow training checkpoint.
            Defaults to `None` if `checkpoint_dir` is not passed.
        ckpt_manager (:obj:`CheckpointManager`, optional): A checkpoint manager, used to save
            :attr:`ckpt` to file.
            Defaults to `None` if `checkpoint_dir` is not passed.
        gp_prior (:obj:`GaussianProcess`): A Gaussian process using :attr:`kernel` and
            using :attr:`observation_index_points` as indices.

    """

    def __init__(
        self,
        observation_index_points: tf.Tensor,
        observations: tf.Tensor,
        checkpoint_dir: Optional[Union[str, Path]] = None,
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

        self.loss = tf.Variable(
            np.nan, dtype=tf.float64, trainable=False, name="training_nll",
        )

        self.metrics = {
            "nll": tf.Variable(
                np.nan, dtype=tf.float64, trainable=False, name="validation_nll",
            ),
            "mae": tf.Variable(
                np.nan, dtype=tf.float64, trainable=False, name="validation_mae",
            ),
            "sharpness": tf.Variable(
                np.nan, dtype=tf.float64, trainable=False, name="validation_sharpness",
            ),
            "variation": tf.Variable(
                np.nan,
                dtype=tf.float64,
                trainable=False,
                name="validation_coeff_variance",
            ),
            "calibration_err": tf.Variable(
                np.nan,
                dtype=tf.float64,
                trainable=False,
                name="validation_calibration_error",
            ),
        }

        if checkpoint_dir:
            self.ckpt = tf.train.Checkpoint(
                step=self.training_steps,
                amp=self.amplitude,
                ls=self.length_scale,
                loss=self.loss,
                val_nll=self.metrics["nll"],
                val_mae=self.metrics["mae"],
                val_sharpness=self.metrics["sharpness"],
                val_coeff_var=self.metrics["variation"],
                val_cal_err=self.metrics["calibration_err"],
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
        """Load a `GPTrainer` model from a file.

        Args:
            model_dir (str): The directory to import the model from.

        Returns:
            The model as a TensorFlow AutoTrackable object.

        """
        return tf.saved_model.load(model_dir)

    def get_model(
        self, index_points: tf.Tensor
    ) -> tfp.python.distributions.GaussianProcessRegressionModel:
        """Get a regression model for a set of index points.

        Args:
            index_points (:obj:`tf.Tensor`): The index points to fit
                regression model.

        Returns:
            gprm (:obj:`GaussianProcessRegressionModel`): The regression model.

        """
        return tfd.GaussianProcessRegressionModel(
            kernel=self.kernel,
            index_points=index_points,
            observation_index_points=self.observation_index_points,
            observations=self.observations,
        )

    @tf.function(input_signature=[tf.TensorSpec(None, tf.float64)])
    def predict(self, points: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Predict targets and the standard deviation of the distribution.

        Args:
            points (:obj:`tf.Tensor`): The points (_x_ values) to make predictions with.

        Returns:
            mean (:obj:`tf.Tensor`): The mean of the distribution at each point.
            stddev (:obj:`tf.Tensor`): The standard deviation of the distribution
                at each point.

        """
        gprm = self.get_model(points)
        return gprm.mean(), gprm.stddev()

    def train_model(
        self,
        val_points: tf.Tensor,
        val_obs: tf.Tensor,
        epochs: int = 1000,
        patience: Optional[int] = None,
        save_dir: Optional[Union[str, Path]] = None,
        metrics: List[str] = [],
    ) -> Iterator[Dict[str, float]]:
        """Optimize model parameters.

        Args:
            val_points (:obj:`tf.Tensor`): The validation points.
            val_obs (:obj:`tf.Tensor`): The validation targets.
            epochs (int): The number of training epochs.
            patience (int, optional): The number of epochs after which to
                stop training if no improvement is seen on the loss of the
                validation data.
            save_dir (str or :obj:`Path`, optional): Where to save the model.
            metrics (list of str): A list of valid metrics to calculate.
                Possible valid metrics are given in :class:`GPMetrics`.

        Yields:
            metrics (dict of str: float): A dictionary of the metrics after the
                last training epoch.

        """
        best_val_nll: float = self.metrics["nll"].numpy()
        if np.isnan(best_val_nll):
            # Set to infinity so < logic works
            best_val_nll = np.inf

        if (self.ckpt_manager or patience) and "nll" not in metrics:
            # We need to track NLL for these to work
            metrics.append("nll")

        steps_since_improvement: int = 1
        gp_metrics = MetricAnalyser(val_points, val_obs, self.get_model(val_points))

        for i in tqdm(range(epochs), "Training epochs"):
            self.loss.assign(self.optimize_cycle())
            self.training_steps.assign_add(1)

            # * Determine and assign metrics
            if gp_metrics.REQUIRES_MEAN.intersection(metrics):
                gp_metrics.update_mean()
            if gp_metrics.REQUIRES_STDDEV.intersection(metrics):
                gp_metrics.update_stddevs()

            try:
                metric_dict: Dict[str, float] = {
                    metric: getattr(gp_metrics, metric) for metric in metrics
                }
            except AttributeError as e:
                raise ValueError(f"Invalid metric: {e}")

            for metric, value in metric_dict.items():
                self.metrics[metric].assign(value)

            metric_dict["loss"] = self.loss.numpy()
            yield metric_dict

            if patience or self.ckpt_manager:
                if self.metrics["nll"] < best_val_nll:
                    best_val_nll = self.metrics["nll"].numpy()
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
