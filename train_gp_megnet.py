"""Utilities for training a GP fed from the MEGNet Concatenation layer for a pretrained model."""
from typing import Optional
from operator import itemgetter

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm

from data_processing import GPDataParser

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


class GPTrainer:
    """Class for training hyperparameters for GP kernels."""

    def __init__(
        self, observation_index_points: tf.Tensor, observations: tf.Tensor,
    ):
        """Initialze attributes and kernel, then optimize."""
        self.observation_index_points = observation_index_points
        self.observations = observations

        # TODO: Customizable kernel
        self.kernel = tfk.MaternOneHalf(
            amplitude=tf.Variable(1.0, dtype=tf.float64, name="amplitude"),
            length_scale=tf.Variable(1.0, dtype=tf.float64, name="length_scale"),
            feature_ndims=self.observation_index_points.shape[1],
        )

        self.gp_prior = tfd.GaussianProcess(self.kernel, self.observation_index_points)
        self.optimizer = tf.optimizers.Adam()

    def get_model(self, index_points: tf.Tensor) -> tfd.GaussianProcessRegressionModel:
        """Get a regression model for a set of index points."""
        return tfd.GaussianProcessRegressionModel(
            kernel=self.kernel,
            index_points=index_points,
            observation_index_points=self.observation_index_points,
            observations=self.observations,
        )

    def train(self, epochs: int = 1000):
        """Optimize the parameters.
        
        Args:
            epochs (int): The number of training steps to perform. Defaults to 1000.

        """
        for i in tqdm(range(epochs), "Training epochs"):
            neg_log_likelihood = self.optimize_cycle()

        print(f"Final NLL after {epochs} steps: {neg_log_likelihood}")

    @tf.function
    def optimize_cycle(self) -> tf.Tensor:
        """Perform one training step.

        Returns:
            loss (:obj:`Tensor`): A Tensor containing the negative log probability loss.

        """
        with tf.GradientTape() as tape:
            loss = -self.gp_prior.log_prob(self.observations, name="log_prob")

        grads = tape.gradient(loss, self.gp_prior.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.gp_prior.trainable_variables))
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

    # Build cation SSE GP model
    cat_gp_trainer = GPTrainer(observation_index_points, cat_observations)
    cat_gp_trainer.train()
    cat_gprm = cat_gp_trainer.get_model(index_points)
