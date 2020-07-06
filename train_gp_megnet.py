"""Utilities for training a GP fed from the MEGNet Concatenation layer for a pretrained model."""
import typing
from operator import itemgetter

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

from data_processing import GPDataParser

tfd = tfp.distributions
tfk = tfp.math.psd_kernels


def convert_index_points(array: np.ndarray) -> tf.Tensor:
    """Reshape an array into a tensor appropriate for GP index points.

    Extends the amount of dimensions by `array.shape[1] - 1` and converts
    to a `Tensor`.

    Args:
        array (:obj:`np.ndarray`): The array to extend.

    Returns
        tensor (:obj:`tf.Tensor`): The converted Tensor.

    """
    shape = array.shape
    shape += (1,) * (shape[1] - 1)
    return tf.constant(array, shape=shape)


class kernel_trainer:
    """Class for training hyperparameters for GP kernels."""

    def __init__(
        self, observation_index_points: tf.constant, observations: tf.constant
    ):
        pass


if __name__ == "__main__":
    # * Get the data and perform some regression
    NDIMS = 96

    train_df = pd.read_pickle("dataframes/gp_train_df.pickle")
    test_df = pd.read_pickle("dataframes/gp_test_df.pickle")

    # Slightly arbitrary kernel choice for now...
    kernel = tfk.MaternOneHalf(feature_ndims=NDIMS)

    observation_index_points = np.stack(train_df["layer_out"].values)
    index_points = np.stack(test_df["layer_out"].values)

    observation_index_points = convert_index_points(observation_index_points)
    index_points = convert_index_points(index_points)

    cation_sses = list(map(itemgetter(0), train_df["sses"]))
    anion_sses = list(map(itemgetter(1), train_df["sses"]))

    cat_observations = tf.constant(cation_sses)
    an_observations = tf.constant(anion_sses)

    # Build cation SSE GP model
    cat_gp_prior = tfd.GaussianProcessRegressionModel(
        kernel=kernel,
        index_points=index_points,
        observation_index_points=observation_index_points,
        observations=cat_observations,
    )
