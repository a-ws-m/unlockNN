"""Utilities for dealing with ``MEGNetModel``s."""
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from megnet.data.crystal import CrystalGraph
from megnet.data.graph import GraphBatchDistanceConvert, GraphBatchGenerator
from megnet.models.megnet import MEGNetModel
from megnet.utils.preprocessing import DummyScaler
from pymatgen.core.structure import Structure

MEGNetGraph = Dict[str, Union[np.ndarray, List[Union[int, float]]]]
Targets = List[Union[float, np.ndarray]]
ModelInput = Union[Structure, MEGNetGraph]


def default_megnet_config(
    nfeat_bond: int = 100, r_cutoff: float = 5.0, gaussian_width: float = 0.5
) -> dict:
    """Get sensible defaults for MEGNetModel configuration.

    These arguments are taken from the `MEGNet README file
    <https://github.com/materialsvirtuallab/megnet#training-a-new-megnetmodel-from-structures>`_.

    Examples:
        Create a MEGNetModel using these defaults:

        >>> model = MEGNetModel(**default_megnet_config())

    """
    gaussian_centres = np.linspace(0, r_cutoff + 1, nfeat_bond)
    graph_converter = CrystalGraph(cutoff=r_cutoff)
    return {
        "graph_converter": graph_converter,
        "centers": gaussian_centres,
        "width": gaussian_width,
    }


def scale_targets(
    meg_model: MEGNetModel, targets: Targets, num_atoms: List[int]
) -> Targets:
    """Scale target values using given MEGNetModel's scaler.

    Args:
        meg_model: The :class:`MEGNetModel` whose scaler to use.
        targets: A list of target values.
        num_atoms: A list of the number of atoms in each structure
            corresponding to the target values.

    Returns:
        The scaled target values.

    """
    return [
        meg_model.target_scaler.transform(target, num_atom)
        for target, num_atom in zip(targets, num_atoms)
    ]


def create_megnet_input(
    meg_model: MEGNetModel,
    inputs: List[ModelInput],
    targets: Optional[Targets] = None,
    batch_size: int = 128,
    scrub_failed_structs: bool = False,
    shuffle: bool = True,
) -> Tuple[Union[GraphBatchDistanceConvert, GraphBatchGenerator], List[MEGNetGraph]]:
    """Create generator for model inputs.

    Args:
        meg_model: The :class:`MEGNetModel` whose graph converter to use.
        inputs: The input, either as graphs or structures.
        targets: The input targets, if any.
        batch_size: The batch size for the generator.
        scrub_failed_structures: Whether to discard structures
            that could not be converted to graphs.
        shuffle: Whether the generator should shuffle the order of the
            structure/target pairs.

    Returns:
        The input generator.

        A list of the model input graphs.

    """
    # Make some targets up for compatibility
    has_targets = targets is not None
    target_buffer = targets if has_targets else [0.0] * len(inputs)

    is_struct = isinstance(inputs[0], Structure)
    if is_struct:
        graphs, trunc_targets = meg_model.get_all_graphs_targets(
            inputs, target_buffer, scrub_failed_structs
        )
    else:
        graphs = inputs
        trunc_targets = target_buffer

    # Check dimensions of model against converted graphs
    meg_model.check_dimension(graphs[0])

    # Scale targets if necessary
    if not isinstance(meg_model.target_scaler, DummyScaler) and has_targets:
        num_atoms = [len(graph["atom"]) for graph in graphs]
        trunc_targets = scale_targets(meg_model, trunc_targets, num_atoms)

    inputs = meg_model.graph_converter.get_flat_data(graphs, trunc_targets)
    return (
        meg_model._create_generator(*inputs, batch_size=batch_size, is_shuffle=shuffle),
        graphs,
    )


if __name__ == "__main__":  # pragma: no cover
    import doctest

    doctest.testmod()
