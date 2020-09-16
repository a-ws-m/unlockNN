"""Test layer extracting functionality."""
import numpy as np
import pymatgen as mg
import pytest
from megnet.data.crystal import CrystalGraph

from unlockgnn.datalib import preprocessing as preproc

test_index = 1
cg = CrystalGraph()

cubic_lattice = mg.Lattice.cubic(4.2)
cscl = mg.Structure(cubic_lattice, ["Cs", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])


@pytest.fixture
def layer_extractor(mocker):
    """Create a layer_extractor instance with a mock model."""
    mock_model = mocker.MagicMock()
    mock_model.graph_converter = cg
    mocker.patch("tensorflow.keras.backend.function")
    return preproc.LayerExtractor(mock_model, test_index)


def test_extractor_init(layer_extractor):
    """Test the initialization procedure for the `LayerExtractor`."""
    layer_extractor.model.layers.__getitem__.assert_called_with(test_index)
    assert callable(layer_extractor.layer_eval)


def test_struct_conversion(layer_extractor):
    """Test converting a structure to an input."""
    converted = layer_extractor._convert_struct_to_inp(cscl)
    assert isinstance(converted, list)
    assert all(isinstance(x, np.ndarray) for x in converted)


# TODO: Lightweight solution to check shape outputs


def test_get_layer_out(layer_extractor):
    """Test getting layer outputs."""
    layer_extractor.get_layer_output(cscl)
    layer_extractor.layer_eval.assert_called_once()


def test_get_layer_out_graph(layer_extractor):
    """Test getting layer outputs for a graph."""
    cscl_graph = cg.convert(cscl)
    layer_extractor.get_layer_output_graph(cscl_graph)
    layer_extractor.layer_eval.assert_called_once()
