"""Test layer scaling functionality.

Todo:
    * Write :class:`LayerScaler` testing code using mocks,
        or a benchmarking :class:`MEGNetModel`.

"""
import numpy as np
import pandas as pd
import pytest

import sse_gnn.datalib.preprocessing as preproc

eye3 = np.eye(3)
max_elem_tests = [
    # We expect a single array input to
    # lead to return of input
    ([eye3], eye3),
    # Check for two arrays, three arrays, ... and five arrays
    *[([eye3 * (j + 1) for j in range(i)], eye3 * i) for i in range(2, 6)],
    # And now with some mixup between the different elements
    ([eye3, 1 - eye3], np.ones((3, 3))),
]


@pytest.mark.parametrize("test_inp,expected", max_elem_tests)
def test_get_max(test_inp, expected):
    """Test `get_max_elements`."""
    assert np.array_equal(preproc.get_max_elements(test_inp), expected)


@pytest.fixture
def trained_layer_scaler(mocker):
    """Get a `LayerScaler` with a mock extractor."""
    le = mocker.Mock()  # Layer extractor mock
    # TODO: This function is WIP
    le.get_layer_output.return_value = None
    mocker.patch("sse_gnn.datalib.preprocessing.LayerExtractor")
    model = mocker.Mock()
