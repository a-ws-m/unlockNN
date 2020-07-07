"""Utilities for training a GP fed from the MEGNet Concatenation layer for a pretrained model."""
from operator import itemgetter
from typing import Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.python.util.deprecation as deprecation
import tensorflow_probability as tfp
from tqdm import tqdm

from data_processing import GPDataParser

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

        self.mae = tf.Variable(np.inf, dtype=tf.float64, trainable=False, name="MAE")

        if checkpoint_dir:
            self.ckpt = tf.train.Checkpoint(
                step=self.training_steps,
                amp=self.amplitude,
                ls=self.length_scale,
                mae=self.mae,
            )
            self.ckpt_manager = tf.train.CheckpointManager(
                self.ckpt,
                checkpoint_dir,
                max_to_keep=3,
                step_counter=self.training_steps,
                # checkpoint_interval=50,
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
        """Predict values for given `points` and the absolute uncertainty as a number of standard deviations."""
        gprm = self.get_model(points)
        prediction = gprm.mean()
        uncertainty = gprm.stddev() * 3.0
        return prediction, uncertainty

    def train_model(
        self,
        val_points: tf.Tensor,
        val_obs: tf.Tensor,
        epochs: int = 1000,
        save_dir: Optional[str] = None,
    ) -> Iterator[float]:
        """Optimize the parameters and measure MAE of validation predictions at each step."""
        best_mae: float = self.mae.numpy()
        for i in tqdm(range(epochs), "Training epochs"):
            neg_log_likelihood = self.optimize_cycle()
            self.training_steps.assign_add(1)
            gprm = self.get_model(val_points)

            self.mae.assign(tf.losses.mae(val_obs, gprm.mean().numpy()))
            if self.mae < best_mae:
                best_mae = self.mae.numpy()
                if self.ckpt_manager:
                    self.ckpt_manager.save(self.training_steps)

            yield self.mae.numpy()

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


if __name__ == "__main__":
    # * Get the data and perform some regression
    NDIMS = 96

    train_df = pd.read_pickle("dataframes/gp_train_df.pickle")
    test_df = pd.read_pickle("dataframes/gp_test_df.pickle")

    observation_index_points = np.stack(train_df["layer_out"].values)
    index_points = np.stack(test_df["layer_out"].values)

    observation_index_points = convert_index_points(observation_index_points)
    index_points = convert_index_points(index_points)

    cation_sses = list(map(itemgetter(0), train_df["sses"]))
    anion_sses = list(map(itemgetter(1), train_df["sses"]))

    cat_observations = tf.constant(cation_sses, dtype=tf.float64)
    an_observations = tf.constant(anion_sses, dtype=tf.float64)

    cat_test_vals = tf.constant(
        list(map(itemgetter(0), test_df["sses"])), dtype=tf.float64
    )
    an_test_vals = tf.constant(
        list(map(itemgetter(1), test_df["sses"])), dtype=tf.float64
    )

    # Build cation SSE GP model
    cat_gp_trainer = GPTrainer(observation_index_points, cat_observations, "./tf_ckpts")
    maes = list(
        cat_gp_trainer.train_model(index_points, cat_test_vals, save_dir="./saved_gp")
    )

    with open("maes1.csv", "a") as f:
        f.write("\n".join(map(str, maes)) + "\n")
