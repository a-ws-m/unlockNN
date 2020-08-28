"""Tools for processing the SSE data for the Gaussian Process."""
from __future__ import annotations

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import pymatgen
from megnet.models import MEGNetModel
from tensorflow.keras import backend as K
from tqdm.contrib import tmap

from ..utilities import deserialize_array


class LayerExtractor:
    """Wrapper for MEGNet Models to extract layer output.

    Args:
        model (:obj:`MEGNetModel`): The MEGNet model to perform extraction
            upon.
        layer_index (int): The index of the layer within the model to extract.
            Defaults to -4, the index of the concatenation layer.

    Attributes:
        model (:obj:`MEGNetModel`): The MEGNet model to perform extraction
            upon.
        conc_layer_output (:obj:`Tensor`): The layer's unevaluated output.
        layer_eval: A Keras function for evaluating the layer.

    """

    def __init__(self, model: MEGNetModel, layer_index: int = -4):
        """Initialize extractor."""
        self.model = model
        self.conc_layer_output = model.layers[layer_index].output
        self.layer_eval = K.function([model.input], [self.conc_layer_output])

    def _convert_struct_to_inp(self, structure: pymatgen.Structure) -> List:
        """Convert a pymatgen structure to an appropriate input for the model.

        Uses MEGNet's builtin methods for doing so.

        Args:
            structure (:obj:`pymatgen.Structure`): A Pymatgen structure
                to convert to an input.

        Returns:
            input (list of :obj:`np.ndarray`): The processed input, ready for
                feeding into the model.

        """
        graph = self.model.graph_converter.convert(structure)
        return self.model.graph_converter.graph_to_input(graph)

    def get_layer_output(self, structure: pymatgen.Structure) -> np.ndarray:
        """Get the layer output for the model.

        Args:
            structure (:obj:`pymatgen.Structure`):
                Pymatgen structure to calculate the layer output for.

        Returns:
            np.ndarray: The output of the layer.

        """
        input = self._convert_struct_to_inp(structure)
        return self.layer_eval([input])[0]

    def get_layer_output_graph(self, graph: Dict[str, np.ndarray]) -> np.ndarray:
        """Get the layer output of the model for a graph.

        Args:
            graph (list):
                MEGNet compatible graph to calculate the layer output for.

        Returns:
            np.ndarray: The output of the layer.

        """
        input = self.model.graph_converter.graph_to_input(graph)
        return self.layer_eval([input])[0]


class LayerScaler:
    """Class for creating GP training data and preprocessing thereof.

    A `LayerScaler` must be initialized with either training data or a
    precalculated scaling factor, if one has already been calculated.
    See :meth:`_calc_scaling_factor` for the procedure for calculating the
    scaling factor.

    Args:
        model: The MEGNet model to perform extraction
            upon.
        sf: The scaling factor.
        layer_index: The index of the layer of the model to extract.
            Unused if extractor is passed.
            Defaults to the concatenation layer index of -4.
        extractor: The :obj:`LayerExtractor` to use for extraction.

    Attributes:
        model: The MEGNet model to perform extraction
            upon.
        sf: The scaling factor.
        extractor: The extractor object used for acquiring layer
            outputs for the model.

    """

    def __init__(
        self,
        model: MEGNetModel,
        sf: np.ndarray,
        layer_index: int = -4,
        extractor: Optional[LayerExtractor] = None,
    ):
        """Initialize class attributes."""
        self.extractor = extractor if extractor else LayerExtractor(model, layer_index)
        self.sf = sf

    @staticmethod
    def from_train_data(
        model: MEGNetModel,
        train_structs: Optional[List[pymatgen.Structure]] = None,
        train_graphs: Optional[List[Dict[str, np.ndarray]]] = None,
        layer_index: int = -4,
    ) -> LayerScaler:
        """Create a LayerScaler instance with a scaling factor based on training data.

        Args:
            train_structs: Training data as structures.
            train_graphs: Training data as graphs.
            layer_index: The index of the layer to extract from.
                Defaults to extraction from the concatenation layer.

        Returns:
            A :obj:`LayerScaler` object.

        Raises:
            ValueError: If neither or both of `train_structs` and `train_graphs` are
                provided.

        """
        ts_given = train_structs is not None
        tg_given = train_graphs is not None

        if ts_given and tg_given:
            raise ValueError("May only pass one of `train_structs` and `train_graphs`")

        extractor = LayerExtractor(model, layer_index)

        if ts_given:
            layer_outs = LayerScaler._calc_layer_outs(train_structs, extractor)  # type: ignore
        elif tg_given:
            layer_outs = LayerScaler._calc_layer_outs(
                train_graphs, extractor, use_structs=False  # type: ignore
            )
        else:
            raise ValueError("Must pass one of `train_structs` or `train_graphs`")

        sf = LayerScaler._calc_scaling_factor(layer_outs)
        return LayerScaler(model, sf, extractor=extractor)

    def structures_to_input(
        self, structures: List[pymatgen.Structure]
    ) -> List[np.ndarray]:
        """Convert structures to a scaled input feature vector.

        Args:
            structures (list of :obj:`pymatgen.Structure`): The structures to convert.

        Returns:
            list of :obj:`np.ndarray`: The scaled feature vectors.

        """
        return [
            out / self.sf for out in self._calc_layer_outs(structures, self.extractor)
        ]

    def graphs_to_input(self, graphs: List[Dict[str, np.ndarray]]) -> List[np.ndarray]:
        """Convert graphs to a scaled input feature vector.

        Args:
            structures (list of dict): The graphs to convert.

        Returns:
            list of :obj:`np.ndarray`: The scaled feature vectors.

        """
        return [
            out / self.sf
            for out in self._calc_layer_outs(graphs, self.extractor, use_structs=False)
        ]

    @staticmethod
    def _calc_layer_outs(
        data: List[Union[pymatgen.Structure, Dict[str, np.ndarray]]],
        extractor: LayerExtractor,
        use_structs: bool = True,
        show_pbar: bool = False,
    ) -> List[np.ndarray]:
        """Calculate the layer outputs for all structures in a list.

        Args:
            data (list of :obj:`pymatgen.Structure` or list of dict): The inputs to calculate
                the layer output for.
            use_structs (bool): Whether `data` are structures (`True`) or graphs (`False`).
            show_pbar: Whether to show a progress bar during calculation of layer outputs.

        Returns:
            layer_outs (list of :obj:`np.ndarray`): The layer outputs.

        Raises:
            TypeError: If `data` contains incompatible types.

        """
        map_f = tmap if show_pbar else map

        if use_structs:
            if not all(isinstance(d, pymatgen.Structure) for d in data):
                raise TypeError("`data` must be a list of structures")
            layer_outs = map_f(extractor.get_layer_output, data)
        else:
            if not all(isinstance(d, dict) for d in data):
                raise TypeError("`data` must be a list of dictionaries")
            layer_outs = map_f(extractor.get_layer_output_graph, data)

        # Squeeze each value to a nicer shape
        return list(map(np.squeeze, layer_outs))

    @staticmethod
    def _calc_scaling_factor(layer_outs: List[np.ndarray]) -> np.ndarray:
        """Calculate the scaling factor to use.

        Scaling factor is the elementwise greatest value across
        all of the `layer_out` vectors.

        Args:
            layer_outs: The layer outputs.

        Returns:
            sf (:obj:`np.ndarray`): The scaling factor.

        """
        abs_values = list(map(np.abs, layer_outs))
        sf = get_max_elements(abs_values)

        # Replace zeros with a scaling factor of 1
        # so there's no zero division errors
        sf[sf == 0.0] = 1.0
        return sf


def get_max_elements(arrays: List[np.ndarray]) -> np.ndarray:
    """Get the elementwise greatest value across all arrays.

    Input arrays must all have the same shape.

    Args:
        arrays (list of :obj:`np.ndarray`): The arrays to perform the calculation with.

    Returns:
        max_array (:obj:np.ndarray): An array of the elementwise max values.

    Raises:
        ValueError: If input arrays do not all have the same shape.

    """
    shapes = set(map(np.shape, arrays))
    if len(shapes) != 1:
        raise ValueError("Input arrays must have uniform shape.")

    stack = np.stack(arrays)
    return np.amax(stack, axis=0)


def convert_graph_df(df: pd.DataFrame) -> List[Dict[str, np.ndarray]]:
    """Convert graph input columns in a DataFrame to a list format.

    DataFrame columns are expected to contain serialized data.

    """
    # Column labels of array data
    array_cols = ["index1", "index2", "atom", "bond", "state"]
    graphs = [
        {col: deserialize_array(series[col]) for col in array_cols}
        for _, series in df.iterrows()
    ]
    print(f"{graphs[0]=}")
    return graphs
