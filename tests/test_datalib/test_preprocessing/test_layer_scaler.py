"""Test layer scaling functionality.

Todo:
    * Write :class:`LayerScaler` testing code using mocks,
        or a benchmarking :class:`MEGNetModel`.

"""
import numpy as np
import pandas as pd
import pymatgen
import pytest
from megnet.models import MEGNetModel

import unlockgnn.datalib.preprocessing as preproc

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


"""Mock fixtures

These are used to mock `LayerExtractor` so that
its `get_layer_output` method can be configured to return the
desired value for a given, mocked `pymatgen.Strucuture`.

"""


@pytest.fixture
def mock_model(mocker):
    """Patch and create a mocked `MEGNetModel`."""
    return mocker.patch("megnet.models.MEGNetModel")


@pytest.fixture
def mock_structure(mocker, request):
    """Get a mock pymatgen Structure.

    The request parameter indicates the `layer_output` it
    should have when run through the mock `LayerExtractor`.

    """
    mock_structure = mocker.Mock(pymatgen.Structure)
    mock_structure.layer_output = request.param
    return mock_structure


@pytest.fixture
def mock_layer_scaler(mocker, mock_model, request):
    """Get a `LayerScaler` with a mock extractor.

    Can be parametrized with a `layer_index`.

    """
    layer_extractor = "unlockgnn.datalib.preprocessing.LayerExtractor"
    mock_le = mocker.patch(layer_extractor, autospec=True)
    mock_le.return_value.get_layer_output = lambda struct: struct.layer_output

    try:
        layer_index = request.param
    except AttributeError:
        # Not passed
        layer_index = None

    return preproc.LayerExtractor(mock_model, layer_index)


"""Test that the mock fixtures work as expected."""


@pytest.mark.parametrize(
    "mock_layer_scaler,mock_structure,expected_layer_output",
    [(None, np.eye(3), np.eye(3))],
    indirect=["mock_layer_scaler", "mock_structure"],
)
def test_layer_scaler_mock(mock_layer_scaler, mock_structure, expected_layer_output):
    """Test that the LayerScaler mock is working."""
    assert np.array_equal(
        mock_layer_scaler.get_layer_output(mock_structure), expected_layer_output
    )


@pytest.mark.parametrize(
    "mock_structure,expected_layer_output",
    [(np.eye(3), np.eye(3))],
    indirect=["mock_structure"],
)
def test_structure_mock(mock_structure, expected_layer_output):
    """Test that the Structure mock is working."""
    assert np.array_equal(mock_structure.layer_output, expected_layer_output)


"""Test `LayerScaler` functionality."""


@pytest.mark.parametrize(
    "sf,mock_structure,expected",
    [
        (np.ones(3), np.eye(3), np.eye(3)),  # Unit scaling factor
        (2 * np.ones(3), np.eye(3), np.eye(3) / 2),
        (
            np.arange(1, 10).reshape((3, 3)),
            np.arange(1, 10).reshape((3, 3)),
            np.ones((3, 3)),
        ),
    ],
    indirect=["mock_structure"],
)
def test_ls_with_sf_no_ex(mock_layer_scaler, mock_structure, mock_model, sf, expected):
    """Test a `LayerScaler` initialized with `sf` but not `extractor`.

    `expected` is the expected scaled layer output. The unscaled layer
    output is the second parameter, which is passed to `mock_structure`.

    """
    ls = preproc.LayerScaler(mock_model, sf)
    scaled = ls.structures_to_input([mock_structure])

    assert len(scaled) == 1
    assert np.array_equal(scaled[0], expected)


@pytest.mark.parametrize(
    "mock_structure,exp_sf", [(np.eye(3), np.ones((3, 3)))], indirect=["mock_structure"]
)
def test_ls_with_structs(mock_layer_scaler, mock_structure, mock_model, exp_sf):
    """Test a `LayerScaler` initialized with training structures."""
    ls = preproc.LayerScaler.from_train_data(mock_model, train_structs=[mock_structure])
    assert np.array_equal(ls.sf, exp_sf)
