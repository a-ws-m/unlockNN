"""Wrappers for main functionality of model training and saving."""
from pathlib import Path
from typing import Literal, List, Optional, Union

import numpy as np
import pymatgen
from megnet.models import MEGNetModel

from .datalib.preprocessing import LayerScaler
from .gp.gp_trainer import GPTrainer
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
        **kwargs,
    ):
        """Initialize `MEGNetModel` and type of GP to use."""
        if gp_type not in ["GP", "VGP"]:
            raise ValueError(f"`gp_type` must be one of 'GP' or 'VGP', got {gp_type=}")
        if gp_type == "GP" and ntarget != 1:
            raise NotImplementedError(
                f"Can only have `ntarget > 1` when `gp_type` is 'VGP' (got {ntarget=})"
            )

        self.gp_type = gp_type
        self.meg_model = MEGNetModel(ntarget=ntarget, **kwargs)
        self.train_structs = train_structs
        self.train_targets = train_targets
        self.val_structs = val_structs
        self.val_targets = val_targets

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

    def train_uq(self, *args, **kwargs):
        """Train the uncertainty quantifier.

        Extracts chosen layer outputs from :attr:`meg_model`,
        scale them and train the appropriate GP (from :attr:`gp_type`).

        """
        pass

    def _train_gp(self, *args, **kwargs):
        """Train a GP on preprocessed layer outputs from a model."""
        pass

    def _train_vgp(self, *args, **kwargs):
        """Train a VGP on preprocessed layer outputs from a model."""
        pass

    def save(self, fname: Union[Path, str]):
        """Save the full-stack model."""
        raise NotImplementedError()

    @staticmethod
    def load(fname: Union[Path, str]):
        """Load a full-stack model."""
        raise NotImplementedError()
