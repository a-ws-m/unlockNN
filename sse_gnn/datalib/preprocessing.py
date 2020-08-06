"""Tools for processing the SSE data for the Gaussian Process."""
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import pymatgen
from megnet.models import MEGNetModel
from tensorflow.keras import backend as K

from sse_gnn.utilities import deserialize_array


class LayerExtractor:
    """Wrapper for MEGNet Models to extract layer output.

    Args:
        model (:obj:`MEGNetModel`): The MEGNet model to perform extraction
            upon.
        layer_index (int): The index of the layer within the model to extract.

    Attributes:
        model (:obj:`MEGNetModel`): The MEGNet model to perform extraction
            upon.
        conc_layer_output (:obj:`Tensor`): The layer's unevaluated output.
        layer_eval: A Keras function for evaluating the layer.

    """

    def __init__(self, model: MEGNetModel, layer_index: int):
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


class ConcatExtractor(LayerExtractor):
    """Wrapper for a `LayerExtractor` that acquires the concatenation layer output."""

    def __init__(self, model: MEGNetModel):
        """Initialize LayerExtractor with concatenation layer index."""
        super().__init__(model, layer_index=-4)


class LayerScaler:
    """Class for creating GP training data and preprocessing thereof.

    A `LayerScaler` must be initialized with either training data or a
    precalculated scaling factor, if one has already been calculated.
    See :meth:`_calc_scaling_factor` for the procedure for calculating the
    scaling factor.

    Args:
        model (:obj:`MEGNetModel`): The MEGNet model to perform extraction
            upon.
        sf (:obj:`np.ndarray`, optional): The scaling factor. Must be passed if
            `training_data` is not passed.
        training_df (:obj:`pd.DataFrame`, optional): The training dataframe to
            use.
        layer_index (int, optional): The index of the layer of the model to extract.
            If unassigned, defaults to extracting from the concatenation layer.
        use_structs (bool): Whether to use structures from the `training_df`. If `False`,
            uses graph inputs.

    Attributes:
        extractor (:obj:`LayerExtractor`): The extractor object used for acquiring layer
            outputs for the model.
        model (:obj:`MEGNetModel`): The MEGNet model to perform extraction
            upon.
        sf (:obj:`np.ndarray`): The scaling factor. Either calculated from
            `training_data` (if passed) or passed as a parameter during initialization.
        training_data (:obj:`pd.DataFrame`, optional): The training dataframe, from which
            the :attr:`sf` is calculated.


    Raises:
        ValueError: If both `training_data` and `sf`, or neither of them, are supplied.

    """

    def __init__(
        self,
        model: MEGNetModel,
        sf: Optional[np.ndarray] = None,
        training_df: Optional[pd.DataFrame] = None,
        layer_index: Optional[int] = None,
        use_structs: bool = True,
    ):
        """Initialize class attributes."""
        self.extractor: LayerExtractor = (
            ConcatExtractor(model)
            if layer_index is None
            else LayerExtractor(model, layer_index)
        )

        if training_df is None:
            self.training_data: Optional[pd.DataFrame] = None
            if sf is None:
                raise ValueError(
                    "Must supply a scaling factor if training data is not supplied."
                )
            self.sf: np.ndarray = sf

        else:
            if sf is not None:
                raise ValueError("Supply only one of training data and scaling factor.")
            self.training_data = training_df.copy()

            # Calculate layer outputs
            if use_structs:
                structures = [
                    pymatgen.Structure.from_str(struct, "json")
                    for struct in self.training_data["structure"]
                ]
                self.training_data["layer_out"] = self._calc_layer_outs(structures)
            else:
                graphs = convert_graph_df(self.training_data)
                self.training_data["layer_out"] = self._calc_layer_outs(
                    graphs, use_structs=False
                )

            # Calculate scaling factor
            self.sf = self._calc_scaling_factor()

            # Scale by the factor
            self.training_data["layer_out"] = self.training_data["layer_out"].apply(
                lambda x: x / self.sf
            )

    def structures_to_input(
        self, structures: List[pymatgen.Structure]
    ) -> List[np.ndarray]:
        """Convert structures to a scaled input feature vector.

        Args:
            structures (list of :obj:`pymatgen.Structure`): The structures to convert.

        Returns:
            list of :obj:`np.ndarray`: The scaled feature vectors.

        """
        return [out / self.sf for out in self._calc_layer_outs(structures)]

    def graphs_to_input(self, graphs: List[Dict[str, np.ndarray]]) -> List[np.ndarray]:
        """Convert graphs to a scaled input feature vector.

        Args:
            structures (list of dict): The graphs to convert.

        Returns:
            list of :obj:`np.ndarray`: The scaled feature vectors.

        """
        return [
            out / self.sf for out in self._calc_layer_outs(graphs, use_structs=False)
        ]

    def _calc_layer_outs(
        self,
        data: List[Union[pymatgen.Structure, Dict[str, np.ndarray]]],
        use_structs: bool = True,
    ) -> List[np.ndarray]:
        """Calculate the layer outputs for all structures in a list.

        Args:
            data (list of :obj:`pymatgen.Structure` or list of dict): The inputs to calculate
                the layer output for.
            use_structs (bool): Whether `data` are structures (`True`) or graphs (`False`).

        Returns:
            layer_outs (list of :obj:`np.ndarray`): The layer outputs.

        Raises:
            TypeError: If `data` contains incompatible types.

        """
        if use_structs:
            if not all(isinstance(d, pymatgen.Structure) for d in data):
                raise TypeError("`data` must be a list of structures")
            layer_outs = map(self.extractor.get_layer_output, data)
        else:
            if not all(isinstance(d, dict) for d in data):
                raise TypeError("`data` must be a list of dictionaries")
            layer_outs = map(self.extractor.get_layer_output_graph, data)

        # Squeeze each value to a nicer shape
        return list(map(np.squeeze, layer_outs))

    def _calc_scaling_factor(self) -> np.ndarray:
        """Calculate the scaling factor to use.

        Scaling factor is the elementwise greatest value across
        all of the `layer_out` vectors.

        Returns:
            sf (:obj:`np.ndarray`): The scaling factor.

        """
        if self.training_data is None:
            raise AttributeError(
                "GPDataParser must have training data assigned"
                " to calculate a scaling factor."
            )

        abs_values = list(map(np.abs, self.training_data["layer_out"]))
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
