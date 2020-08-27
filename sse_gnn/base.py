"""Wrappers for main functionality of model training and saving."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Literal, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pymatgen
import tensorflow as tf
from megnet.models import MEGNetModel
from pyarrow import feather

from .datalib.preprocessing import LayerScaler
from .gp.gp_trainer import GPTrainer, convert_index_points
from .gp.vgp_trainer import SingleLayerVGP
from .utilities.serialization import deserialize_array, serialize_array


class MEGNetProbModel:
    """A base MEGNetModel with uncertainty quantification.

    Args:
        train_structs: The training structures.
        train_targets: The training targets.
        val_structs: The validation structures.
        val_targets: The validation targets.
        gp_type: The method to use for the Gaussian process.
            Must be either 'GP' or 'VGP'.
        save_dir: The directory to save files to during training.
            Files include MEGNet and GP checkpoints.
        ntarget: The number of target variables.
            This can only be greater than one if `gp_type` is 'VGP'.
        layer_index: The index of the layer to extract outputs from
            within :attr:`meg_model`. Defaults to the concatenation
            layer.
        num_inducing_points: The number of inducing points for the `VGP`.
            Can only be set for `gp_type='VGP'`.
        training_stage: The stage of training the model is at.
            Only applies when loading a model.
        **kwargs: Keyword arguments to pass to :class:`MEGNetModel`.

    """

    def __init__(
        self,
        train_structs: List[pymatgen.Structure],
        train_targets: List[Union[np.ndarray, float]],
        val_structs: List[pymatgen.Structure],
        val_targets: List[Union[np.ndarray, float]],
        gp_type: Literal["GP", "VGP"],
        save_dir: Union[str, Path],
        ntarget: int = 1,
        layer_index: int = -4,
        num_inducing_points: Optional[int] = None,
        training_stage: int = 0,
        **kwargs,
    ):
        """Initialize `MEGNetModel` and type of GP to use."""
        if gp_type not in ["GP", "VGP"]:
            raise ValueError(f"`gp_type` must be one of 'GP' or 'VGP', got {gp_type=}")
        if gp_type == "GP":
            if ntarget > 1:
                raise NotImplementedError(
                    f"Can only have `ntarget > 1` when `gp_type` is 'VGP' (got {ntarget=})"
                )
            if num_inducing_points is not None:
                raise ValueError(
                    "`num_inducing_points` can only be set when `gp_type` is `VGP`"
                )
        if gp_type == "VGP":
            if num_inducing_points is None:
                raise ValueError(
                    "`num_inducing_points` must be supplied for `gp_type=VGP`, "
                    f"got {num_inducing_points=}"
                )

        self.gp_type = gp_type
        self.train_structs = train_structs
        self.train_targets = train_targets
        self.val_structs = val_structs
        self.val_targets = val_targets

        self.save_dir = Path(save_dir)
        self.ntarget = ntarget
        self.sf: Optional[np.ndarray] = None
        self.layer_index = layer_index
        self.num_inducing_points = num_inducing_points

        self.meg_ckpt_path = self.save_dir / "meg_ckpts"
        self.meg_save_path = self.save_dir / "meg_model"
        self.gp_ckpt_path = self.save_dir / "gp_ckpts"
        self.gp_save_path = self.save_dir / "gp_model"

        self.data_save_path = self.save_dir / "data"
        self.train_database = self.data_save_path / "train.fthr"
        self.val_database = self.data_save_path / "val.fthr"
        self.sf_path = self.data_save_path / "sf"
        self.meta_path = self.data_save_path / "meta.txt"

        # * Make directories
        for direct in [self.save_dir, self.data_save_path]:
            os.makedirs(direct, exist_ok=True)

        self.meg_model = (
            MEGNetModel(ntarget=ntarget, **kwargs)
            if training_stage == 0
            else MEGNetModel.from_file(str(self.meg_save_path))
        )

        # Initialize GP
        if training_stage < 2:
            self.gp: Optional[Union[GPTrainer, SingleLayerVGP]] = None
        else:
            index_points = np.stack(self.get_index_points(self.train_structs))

            if gp_type == "VGP":
                index_points = tf.constant(index_points, dtype=tf.float64)
                # Should already have been caught, but for the type checker's sake
                assert num_inducing_points is not None
                self.gp = SingleLayerVGP(
                    index_points,
                    num_inducing_points,
                    ntarget,
                    prev_model=str(self.gp_save_path),
                )

            else:
                index_points = convert_index_points(index_points)
                targets = tf.constant(np.stack(self.train_targets), dtype=tf.float64)
                self.gp = GPTrainer(index_points, targets, self.gp_ckpt_path)

    @property
    def training_stage(self) -> Literal[0, 1, 2]:
        """Indicate the training stage the model is at.

        Returns:
            training_stage: How much of the model is trained.
                Can take one of three values:

                * 0 - Untrained.
                * 1 - :attr:`meg_model` trained.
                * 2 - :attr:`meg_model` and :attr:`gp` trained.

        """
        return int(self.meg_save_path.exists()) + bool(self.gp)  # type: ignore

    def train_meg_model(
        self, epochs: Optional[int] = 1000, batch_size: Optional[int] = 128, **kwargs,
    ):
        """Train the MEGNetModel.

        Args:
            epochs: The number of training epochs.
            batch_size: The batch size.
            **kwargs: Keyword arguments to pass to :func:`MEGNetModel.train`.

        """
        self.meg_model.train(
            self.train_structs,
            self.train_targets,
            self.val_structs,
            self.val_targets,
            epochs,
            batch_size,
            dirname=self.meg_ckpt_path,
            **kwargs,
        )

        self.meg_model.save_model(str(self.meg_save_path))
        self._update_sf()

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
            self.gp, _ = self._train_gp(training_idxs, val_idxs, epochs, **kwargs)
        else:
            training_idxs = tf.constant(training_idxs, dtype=tf.float64)
            val_idxs = tf.constant(val_idxs, dtype=tf.float64)
            self.gp = self._train_vgp(training_idxs, val_idxs, epochs, **kwargs)

    def _train_gp(
        self,
        train_idxs: List[np.ndarray],
        val_idxs: List[np.ndarray],
        epochs: int,
        **kwargs,
    ) -> Tuple[GPTrainer, List[Dict[str, float]]]:
        """Train a GP on preprocessed layer outputs from a model."""
        if self.gp_type != "GP":
            raise ValueError("Can only train GP for `gp_type='GP'`")

        train_targets = tf.constant(np.stack(self.train_targets), dtype=tf.float64)
        val_targets = tf.constant(np.stack(self.val_targets), dtype=tf.float64)

        gp_trainer = GPTrainer(
            train_idxs, train_targets, checkpoint_dir=str(self.gp_ckpt_path)
        )
        metrics = list(
            gp_trainer.train_model(
                val_idxs, val_targets, epochs, save_dir=str(self.gp_save_path), **kwargs
            )
        )
        return gp_trainer, metrics

    def _train_vgp(
        self,
        train_idxs: List[np.ndarray],
        val_idxs: List[np.ndarray],
        epochs: int,
        **kwargs,
    ) -> SingleLayerVGP:
        """Train a VGP on preprocessed layer outputs from a model."""
        if self.gp_type != "VGP":
            raise ValueError("Can only train VGP for `gp_type='VGP'`")
        if self.num_inducing_points is None:
            # This should already have been handled in __init__, but just in case
            raise ValueError("Cannot train VGP without `num_inducing_points`")

        train_targets = targets_to_tensor(self.train_targets)
        val_targets = targets_to_tensor(self.val_targets)

        vgp = SingleLayerVGP(train_idxs, self.num_inducing_points, self.ntarget)
        vgp.train_model(
            train_targets,
            (val_idxs, val_targets),
            epochs,
            checkpoint_path=str(self.gp_ckpt_path),
            **kwargs,
        )
        vgp.model.save_weights(str(self.gp_save_path))
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
        if self.gp is None:
            raise ValueError(
                "UQ must be trained using `train_uq` before making predictions."
            )

        index_point = self.get_index_points([struct])[0]
        index_point = tf.Tensor(index_point, dtype=tf.float64)
        predicted, uncert = self.gp.predict(index_point)
        return predicted.numpy(), uncert.numpy()

    def _validate_id_len(self, ids: Optional[List[str]], is_train: bool):
        """Check that the supplied IDs' length matches the length of saved data.

        Passes by default if `ids is None`.

        Args:
            ids: The IDs to check.
            is_train: Whether the supplied IDs correspond to train data (`True`)
                or validation data (`False`).

        Raises:
            ValueError: If there is a length mismatch.

        """
        if ids is not None:
            id_name = "train" if is_train else "val"
            struct_len = len(self.train_structs if is_train else self.val_structs)

            if (id_len := len(ids)) != struct_len:
                raise ValueError(
                    f"Length of supplied `{id_name}_materials_ids`, {id_len}, "
                    f"does not match length of `{id_name}_structs`, {struct_len}"
                )

    def save(
        self,
        train_materials_ids: Optional[List[str]] = None,
        val_materials_ids: Optional[List[str]] = None,
    ):
        """Save the full-stack model.

        Args:
            train_materials_ids: A list of IDs corresponding to :attr:`train_structs`.
                Used for indexing in the saved database.
            val_materials_ids: A list of IDs corresponding to :attr:`val_structs`.
                Used for indexing in the saved database.

        """
        for validation_args in [
            (train_materials_ids, True),
            (val_materials_ids, False),
        ]:
            self._validate_id_len(*validation_args)

        # * Write training + validation data

        train_data = self._gen_serial_data(self.train_structs, self.train_targets)
        val_data = self._gen_serial_data(self.val_structs, self.val_targets)

        train_df = pd.DataFrame(train_data, train_materials_ids)
        val_df = pd.DataFrame(val_data, val_materials_ids)

        feather.write_feather(train_df, self.train_database)
        feather.write_feather(val_df, self.val_database)

        # * Write metadata
        self._write_metadata()

    def _gen_serial_data(
        self, structs: List[pymatgen.Structure], targets: List[Union[float, np.ndarray]]
    ) -> Dict[str, List[Union[str, float, bytes]]]:
        """Convert a list of structures into a precursor dictionary for a DataFrame."""
        data = {"struct": [struct.to("json") for struct in structs]}

        if self.ntarget > 1:
            data["target"] = [serialize_array(arr) for arr in targets]
        else:
            data["target"] = [
                (target.item() if isinstance(target, np.ndarray) else target)
                for target in targets
            ]

        # ? Currently no need to save index_points
        # if self.training_stage > 0:
        #     data["index_points"] = [
        #         serialize_array(ips) for ips in self.get_index_points(structs)
        #     ]

        return data

    def _write_metadata(self):
        """Write metadata to a file.

        Metadata contains :attr:`gp_type`, :attr:`num_inducing_points`,
        :attr:`layer_index`, :attr:`ntarget` and :attr:`training_stage`.

        """
        meta = {
            "gp_type": self.gp_type,
            "num_inducing_points": self.num_inducing_points,
            "layer_index": self.layer_index,
            "ntarget": self.ntarget,
            "training_stage": self.training_stage,
        }
        with self.meta_path.open("w") as f:
            json.dump(meta, f)

    @staticmethod
    def _load_serial_data(fname: Union[Path, str]) -> pd.DataFrame:
        """Load serialized data.

        The reverse of :meth:`_gen_serial_data`.

        """
        data = feather.read_feather(fname)
        data.loc["struct"] = data["struct"].apply(pymatgen.Structure.from_str)

        if isinstance(data["target"][0], bytes):
            # Data is serialized
            data.loc["target"] = data["target"].apply(deserialize_array)

        # ? index_points not currently saved
        # try:
        #     data.loc["index_points"] = data["index_points"].apply(deserialize_array)
        # except KeyError:
        #     # No index points in dataset
        #     pass

        return data

    @staticmethod
    def load(dirname: Union[Path, str]) -> MEGNetProbModel:
        """Load a full-stack model."""
        save_dir = Path(dirname)
        train_datafile = save_dir / "train.fthr"
        val_datafile = save_dir / "val.fthr"

        train_data = MEGNetProbModel._load_serial_data(train_datafile)
        val_data = MEGNetProbModel._load_serial_data(val_datafile)

        metafile = save_dir / "meta.txt"
        with metafile.open("r") as f:
            meta = json.load(f)

        return MEGNetProbModel(
            train_data["struct"],
            train_data["target"],
            val_data["struct"],
            val_data["target"],
            save_dir=save_dir,
            **meta,
        )


def targets_to_tensor(targets: List[Union[float, np.ndarray]]) -> tf.Tensor:
    """Convert a list of target values to a Tensor."""
    return tf.constant(np.stack(targets), dtype=tf.float64)
