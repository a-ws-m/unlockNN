"""Inducing index points initializers."""
from typing import List, Optional
from megnet.models.megnet import MEGNetModel

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from pymatgen.core.structure import Structure

from .megnet_utils import Targets, create_megnet_input


class SampleInitializer(keras.initializers.Initializer):
    """Initializer that samples index points from training data."""

    def __init__(
        self,
        train_structs: List[Structure],
        meg_model: MEGNetModel,
        num_inducing_points: Optional[int] = None,
        latent_layer: int = -2,
        batch_size: int = 128,
        scrub_failed_structs: bool = False,
    ) -> None:
        """Get index points for structures."""
        self.num_inducing_points = num_inducing_points

        if num_inducing_points is None:
            self.train_structs = train_structs
        else:
            rng = np.random.default_rng()
            self.train_structs = list(
                rng.choice(train_structs, num_inducing_points, replace=False)
            )

        self.meg_model = meg_model

        self.nn = self.meg_model.model

        if isinstance(latent_layer, int):
            self.latent_idx: int = latent_layer
        else:
            self.latent_idx = [layer.name for layer in self.nn.layers].index(
                latent_layer
            )

        self.compute_points(batch_size, scrub_failed_structs)

    def compute_points(self, batch_size: int = 128, scrub_failed_structs: bool = False):
        """Update stored index points."""
        latent_layer = self.nn.layers[self.latent_idx].output
        index_pt_model = keras.Model(self.nn.inputs, latent_layer)

        input_gen, _ = create_megnet_input(
            self.meg_model,
            self.train_structs,
            None,
            batch_size,
            scrub_failed_structs,
            shuffle=False,
        )

        points = []
        for inp in input_gen:
            points.append(index_pt_model.predict(inp[:-1]))

        if len(points) == 1:
            self.points = points[0]
        else:
            self.points = np.concatenate(points, axis=1)

    def __call__(self, shape, dtype):
        """Sample some index points."""
        rng = np.random.default_rng()
        samples = rng.choice(self.points, shape[1], axis=1, replace=False)
        return tf.convert_to_tensor(samples, dtype=dtype)
