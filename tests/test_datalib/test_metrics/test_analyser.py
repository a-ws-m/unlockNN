"""Testing for the `MetricAnalyser` class."""
from typing import List, Tuple

import numpy as np
import pytest
import tensorflow as tf
from numpy.testing import assert_almost_equal

from sse_gnn.datalib.metrics import MetricAnalyser

# * Generate some test data
uni_test_points = np.linspace(-1.0, 1.0)
uni_test_vals = uni_test_points * 1.1

bi_test_points = np.column_stack((uni_test_points, uni_test_points))
bi_test_vals = np.column_stack((uni_test_vals, uni_test_points))

test_points: List[Tuple[np.ndarray, np.ndarray]] = [
    (bi_test_points, bi_test_vals,),  # Bivariate test points
    (uni_test_points, uni_test_vals),  # Univaritate test points
]


@pytest.fixture(params=test_points)
def analyser(request, mocker):
    """Create a `MetricAnalyser` for different distributions.

    The distributions have means centred precisely on target
    values for each index point, with a standard deviation of
    one and a log probability of two, for testing.

    """
    index_points = request.param[0]
    targets = request.param[1]

    dist = mocker.Mock()
    dist.mean.return_value = tf.constant(targets)
    dist.stddev.return_value = tf.constant(np.ones(targets.shape))
    dist.log_prob.return_value = tf.constant(2.0)

    return MetricAnalyser(tf.constant(index_points), tf.constant(targets), dist)


def test_mae(analyser):
    """Test whether MAE is correctly calculated."""
    calc_mae = analyser.mae
    assert isinstance(calc_mae, float)
    # The MAE should be about zero, because the means are centred
    # on the target points
    assert_almost_equal(calc_mae, 0)


def test_sharpness(analyser):
    """Test whether the sharpness is correctly calculated.

    The RMS of a list of ones is one.

    """
    assert_almost_equal(analyser.sharpness, 1)


def test_variation(analyser):
    """Test whether the coefficient of variation is correctly calculated.

    Should be zero when all standard deviations are the same.

    """
    assert_almost_equal(analyser.variation, 0)


def test_residuals(analyser):
    """Test that the residuals are correctly calculated.

    Should all be zero when the mean values are exactly correlated.

    """
    assert_almost_equal(analyser.residuals, np.zeros(analyser.val_obs.shape))


def test_pis(analyser):
    """Test that the percentile intervals are correctly calculated.

    When all the means are exactly correlated, 100% of residuals lie on
    the centre of the standard normal distribution.

    """
    predicted_pi, observed_pi = analyser.pis
    expected_pi = np.zeros(predicted_pi.shape)
    expected_pi[predicted_pi >= 0.5] = 1
    assert_almost_equal(observed_pi, expected_pi)
