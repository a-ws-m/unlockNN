"""Utilities for training a GP fed from the MEGNet Concatenation layer for a pretrained model."""
from typing import List

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
            structure (:obj:`pymatgen.Structure`): A Pymatgen structure
                to evaluate the output for.

        Returns:
            np.ndarray: The output of the concatenation layer, with shape
                (1, 1, 96).

        """
        input = self._convert_struct_to_inp(structure)
        return self.conc_layer_eval([input])[0]


if __name__ == "__main__":
    pass
