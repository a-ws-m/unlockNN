"""Tools for processing the SSE data for the Gaussian Process."""
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import pymatgen
from megnet.models import MEGNetModel
from tensorflow.keras import backend as K


class ConcatExtractor:
    """Wrapper for MEGNet Models to extract concatenation layer output.

    Params:
        model (:obj:`MEGNetModel`): The MEGNet model to perform extraction
            upon.
        conc_layer_output (:obj:`Tensor`): The concatenation layer's
            unevaluated output.
        conc_layer_eval: A Keras function for evaluating the concatenation
            layer.

    """

    def __init__(self, model: MEGNetModel):
        """Initialize extractor."""
        self.model = model
        # -4 is the index of the concatenation layer
        self.conc_layer_output = model.layers[-4].output
        self.conc_layer_eval = K.function([model.input], [self.conc_layer_output])

    def _convert_struct_to_inp(self, structure: pymatgen.Structure) -> List:
        """Convert a pymatgen structure to an appropriate input for the model.

        Uses MEGNet's builtin methods for doing so.

        Args:
            structure (:obj:`pymatgen.Structure`): A Pymatgen structure
                to convert to an input.

        Returns:
            input (list): The processed input, ready for feeding into the
                model.

        """
        graph = self.model.graph_converter.convert(structure)
        return self.model.graph_converter.graph_to_input(graph)

    def get_concat_output(self, structure: pymatgen.Structure) -> np.ndarray:
        """Get the concatenation layer output for the model.

        Args:
            structure (:obj:`pymatgen.Structure`):
                Pymatgen structure to calculate the layer output for.

        Returns:
            np.ndarray: The output of the concatenation layer,
                with shape (1, 1, 96).

        """
        input = self._convert_struct_to_inp(structure)
        return self.conc_layer_eval([input])[0]

    def get_concat_output_graph(self, graph: List) -> np.ndarray:
        """Get the concatenation layer output for the model for a graph.

        Args:
            graph (list):
                MEGNet compatible graph to calculate the layer output for.

        Returns:
            np.ndarray: The output of the concatenation layer,
                with shape (1, 1, 96).

        """
        input = self.model.graph_converter.graph_to_input(graph)
        return self.conc_layer_eval([input])[0]


class GPDataParser:
    """Class for creating GP training data and preprocessing thereof."""

    def __init__(
        self,
        model: MEGNetModel,
        sf: Optional[np.ndarray] = None,
        training_df: Optional[pd.DataFrame] = None,
    ):
        """Initialize class attributes."""
        self.extractor = ConcatExtractor(model)

        if training_df is None:
            self.training_data = None

            if sf is None:
                raise ValueError(
                    "Must supply a scaling factor if training data is not supplied."
                )
            self.sf = sf

        else:

            if sf is not None:
                raise ValueError("Supply only one of training data and scaling factor.")
            self.training_data = training_df.copy()

            # Calculate layer outputs
            structures = [
                pymatgen.Structure.from_str(struct, "json")
                for struct in self.training_data["structure"]
            ]
            self.training_data["layer_out"] = self._calc_layer_outs(structures)

            # Calculate scaling factor
            self.sf = self._calc_scaling_factor()

            # Scale by the factor
            self.training_data["layer_out"] = self.training_data["layer_out"].apply(
                lambda x: x / self.sf
            )

    def structures_to_input(
        self, structures: List[pymatgen.Structure]
    ) -> List[np.ndarray]:
        """Convert structures to a scaled input feature vector."""
        return [out / self.sf for out in self._calc_layer_outs(structures)]

    def _calc_layer_outs(self, data: List[pymatgen.Structure]) -> List[np.ndarray]:
        """Calculate the layer outputs for all structures in a list."""
        layer_outs = map(self.extractor.get_concat_output, data)
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

        abs_values = map(np.abs, self.training_data["layer_out"])

        layer_shape = self.training_data["layer_out"][0].shape
        sf = np.zeros(layer_shape)
        for array in abs_values:
            sf = np.maximum(sf, array)

        # Replace zeros with a scaling factor of 1
        # so there's no zero division errors
        sf[sf == 0.0] = 1.0
        return sf
