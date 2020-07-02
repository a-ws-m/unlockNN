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


def gen_index_points(ndim: int, npoints: int) -> np.ndarray:
    """Generate index points with a given number of dimensions.

    Index points are based off a uniform distribution from -1 to 1.

    Args:
        ndim (int): The number of dimensions.
        npoints (int): The number of samples per dimension.
    
    Returns:
        points_matrix: A matrix with shape (npoints, ndim).
    
    """
    lin_points = np.linspace(-1, 1, npoints)
    points_matrix = np.stack([lin_points for _ in range(ndim)])
    points_matrix = points_matrix.T
    return points_matrix


if __name__ == "__main__":
    # * Get the data and perform some regression
    train_df = pd.read_pickle("dataframes/gp_train_df.pickle")
    test_df = pd.read_pickle("dataframes/gp_test_df.pickle")

    # Slightly arbitrary kernel choice for now...
    kernel = tfk.MaternOneHalf(feature_ndims=96)

    # Create our index points
    index_points = gen_index_points(96, 100)

    # Observation index points
    observation_index_points = np.stack(train_df["layer_out"].values)

    cation_sses = map(itemgetter(0), train_df["sses"])
    anion_sses = map(itemgetter(1), train_df["sses"])

    cat_observations = np.array(cation_sses)[:, np.newaxis]
    an_observations = np.array(anion_sses)[:, np.newaxis]

    # Build cation SSE GP model
    cat_gp_prior = tfd.GaussianProcess(kernel=kernel, index_points=index_points)
