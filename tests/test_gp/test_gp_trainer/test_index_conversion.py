"""Tests for the `convert_index_points` function."""
import numpy as np
import pytest

from unlockgnn.gp.gp_trainer import convert_index_points


ten_space = np.linspace(1, 10, 10)
three_space = np.linspace(1, 3, 3)

# Our test inputs should have shape (n_samples, sample_dimensions)
test_pairs = [
    # First, with ten one-dimensional inputs
    (ten_space, (10, 1)),
    # Now ten three-dimensional inputs
    (np.stack([three_space for _ in range(10)]), (10, 3, 1, 1)),
    # And three three-dimensional inputs
    (np.stack([three_space for _ in range(3)]), (3, 3, 1, 1)),
]


@pytest.mark.parametrize("input,exp_shape", test_pairs)
def test_conversion(input, exp_shape):
    """Test whether points' shapes are correctly altered."""
    converted = convert_index_points(input)
    assert converted.numpy().shape == exp_shape
    assert np.array_equal(converted.numpy().squeeze(), input)
