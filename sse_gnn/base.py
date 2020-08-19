"""Wrappers for main functionality of model training and saving."""
from pathlib import Path
from typing import Literal, List, Optional, Tuple, Union

import numpy as np
import pymatgen
import tensorflow as tf
from megnet.models import MEGNetModel

from .datalib.preprocessing import LayerScaler
from .gp.gp_trainer import GPTrainer, convert_index_points
from .gp.vgp_trainer import SingleLayerVGP
from .utilities.serialization import deserialize_array, serialize_array


class MEGNetProbModel:
    """A base MEGNetModel with uncertainty quantification.

    Args:
        train_structs: The training structures.
        train_targets: The training targets.
        gp_type: The method to use for the Gaussian process.
            Must be either 'GP' or 'VGP'.
        val_structs: The validation structures.
        val_targets: The validation targets.
        ntarget: The number of target variables.
            This can only be greater than one if `gp_type` is 'VGP'.
        layer_index: The index of the layer to extract outputs from
            within :attr:`meg_model`. Defaults to the concatenation
            layer.
        **kwargs: Keyword arguments to pass to :class:`MEGNetModel`.

    """

    def __init__(
        self,
        train_structs: List[pymatgen.Structure],
        train_targets: List[np.ndarray],
        gp_type: Literal["GP", "VGP"],
        val_structs: List[pymatgen.Structure],
        val_targets: List[np.ndarray],
        ntarget: int = 1,
        layer_index: int = -4,
        num_inducing_points: Optional[int] = None,
        **kwargs,
    ):
        """Initialize `MEGNetModel` and type of GP to use."""
        if gp_type not in ["GP", "VGP"]:
            raise ValueError(f"`gp_type` must be one of 'GP' or 'VGP', got {gp_type=}")
        if gp_type == "GP" and ntarget != 1:
            raise NotImplementedError(
                f"Can only have `ntarget > 1` when `gp_type` is 'VGP' (got {ntarget=})"
            )
        if gp_type == "GP" and num_inducing_points is not None:
            raise ValueError(
                "`num_inducing_points` can only be set when `gp_type` is `VGP`"
            )
        if gp_type == "VGP" and num_inducing_points is None:
            raise ValueError(
                "`num_inducing_points` must be supplied for `gp_type=VGP`, "
                f"got {num_inducing_points=}"
            )

        self.gp_type = gp_type
        self.meg_model = MEGNetModel(ntarget=ntarget, **kwargs)
        self.train_structs = train_structs
        self.train_targets = train_targets
        self.val_structs = val_structs
        self.val_targets = val_targets
        self.ntarget = ntarget
        self.sf: Optional[np.ndarray] = None
        self.layer_index = layer_index
        self.gp: Optional[Union[GPTrainer, SingleLayerVGP]] = None
        self.num_inducing_points = num_inducing_points

    def train_meg_model(
        self,
        epochs: Optional[int] = 1000,
        batch_size: Optional[int] = 128,
        save_dir: Optional[Union[Path, str]] = None,
        **kwargs,
    ):
        """Train the MEGNetModel.

        Args:
            epochs: The number of training epochs.
            batch_size: The batch size.
            save_dir: A directory to save the trained MEGNetModel.
            **kwargs: Keyword arguments to pass to :func:`MEGNetModel.train`.

        """
        self.meg_model.train(
            self.train_structs,
            self.train_targets,
            self.val_structs,
            self.val_targets,
            epochs,
            batch_size,
            **kwargs,
        )
        if save_dir:
            self.meg_model.save_model(str(save_dir))

    def _update_sf(self):
        """Update the saved scaling factor.

        This must be called to update :attr:`sf` whenever the MEGNetModel
        is updated (i.e. trained).

        """
        ls = LayerScaler.from_train_data(
            self.meg_model, self.train_structs, layer_index=self.layer_index
        )
        self.sf = ls.sf

    def get_index_points(
        self, structures: List[pymatgen.Structure]
    ) -> List[np.ndarray]:
        """Determine and preprocess index points for GP training.

        Args:
            structures: A list of structrues to convert to inputs.

        Returns:
            index_points: The feature arrays of the structures.

        """
        ls = LayerScaler(self.meg_model, self.sf, self.layer_index)
        return ls.structures_to_input(structures)

    def train_uq(self, epochs: int = 500, **kwargs):
        """Train the uncertainty quantifier.

        Extracts chosen layer outputs from :attr:`meg_model`,
        scale them and train the appropriate GP (from :attr:`gp_type`).

        """
        training_idxs = np.stack(self.get_index_points(self.train_structs))
        val_idxs = np.stack(self.get_index_points(self.val_structs))

        if self.gp_type == "GP":
            training_idxs = convert_index_points(training_idxs)
            val_idxs = convert_index_points(val_idxs)
            self.gp = self._train_gp(training_idxs, val_idxs, epochs)
        else:
            training_idxs = tf.constant(training_idxs, dtype=tf.float64)
            val_idxs = tf.constant(val_idxs, dtype=tf.float64)
            self.gp = self._train_vgp(training_idxs, val_idxs, epochs)

    def _train_gp(
        self,
        train_idxs: List[np.ndarray],
        val_idxs: List[np.ndarray],
        epochs: int,
        **kwargs,
    ):
        """Train a GP on preprocessed layer outputs from a model."""
        if self.gp_type != "GP":
            raise ValueError("Can only train GP for `gp_type='GP'`")

        gp_trainer = GPTrainer(train_idxs, self.train_targets)
        metrics = list(gp_trainer.train_model(val_idxs, self.val_targets, epochs))
        return gp_trainer

    def _train_vgp(
        self,
        train_idxs: List[np.ndarray],
        val_idxs: List[np.ndarray],
        epochs: int,
        **kwargs,
    ):
        """Train a VGP on preprocessed layer outputs from a model."""
        if self.gp_type != "VGP":
            raise ValueError("Can only train VGP for `gp_type='VGP'`")
        if self.num_inducing_points is None:
            # This should already have been handled in __init__, but just in case
            raise ValueError("Cannot train VGP without `num_inducing_points`")

        vgp = SingleLayerVGP(train_idxs, self.num_inducing_points, self.ntarget)
        vgp.train_model(self.train_targets, (val_idxs, self.val_targets), epochs)
        return vgp

    def predict_structure(
        self, struct: pymatgen.Structure
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict target value and an uncertainty for a given structure.

        Args:
            struct: The structure to make predictions on.

        Returns:
            predicted_target: The predicted target value(s).
            uncertainty: The uncertainty in the predicted value(s).

        """
        raise NotImplementedError()

    def save(self, fname: Union[Path, str]):
        """Save the full-stack model."""
        raise NotImplementedError()

    @staticmethod
    def load(fname: Union[Path, str]):
        """Load a full-stack model."""
        raise NotImplementedError()
