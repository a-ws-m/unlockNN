"""Unit tests for serialization module."""
import numpy as np
import pandas as pd
import pytest
from pyarrow import feather

from unlockgnn.utilities.serialization import deserialize_array, serialize_array

test_arrays = [
    # Must test different shapes and datatypes
    np.array([1]),
    np.array([1.0]),
    np.array([1.0, 2.0]),
    np.eye(3),
    np.ones((5, 4)),
    np.ones((3, 2), dtype=np.float64),
]


@pytest.mark.parametrize("test_arr", test_arrays)
def test_reversible_ser(test_arr):
    """Test whether serialization is reversible."""
    serialized = serialize_array(test_arr)
    deserialized = deserialize_array(serialized)
    assert np.array_equal(test_arr, deserialized)


@pytest.mark.parametrize("test_arr", test_arrays)
def test_writable(test_arr, tmp_path):
    """Test that the serialized arrays are writable using `feather`."""
    test_fname = tmp_path / "test_db.fthr"
    serialized = serialize_array(test_arr)
    df = pd.DataFrame({"ser": [serialized]})
    feather.write_feather(df, test_fname)
