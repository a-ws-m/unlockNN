"""Wrappers for main functionality of model training and saving."""
from __future__ import annotations

import json
import os
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
import pandas as pd
import pymatgen
import tensorflow as tf
import tensorflow_probability as tfp
from megnet.data.crystal import CrystalGraph
from megnet.models import MEGNetModel
from pyarrow import feather

from .datalib.preprocessing import LayerScaler
from .gp.gp_trainer import GPTrainer, convert_index_points
from .gp.vgp_trainer import SingleLayerVGP
from .utilities.serialization import deserialize_array, serialize_array

GNN = TypeVar("GNN")


class ProbGNN(ABC):
    """An abstract class for developing GNNs with uncertainty quantification.

    Provides a bundled interface for creating, training, saving and loading
    a GNN and a Gaussian process, minimising data handling for the end user.

    Args:
        train_structs: The training structures.
        train_targets: The training targets.
        val_structs: The validation structures.
        val_targets: The validation targets.
        gp_type: The method to use for the Gaussian process.
            Must be either 'GP' or 'VGP'.
        save_dir: The directory to save files to during training.
            Files include GNN and GP checkpoints.
        ntarget: The number of target variables.
            This can only be greater than one if `gp_type` is 'VGP'.
        layer_index: The index of the layer to extract outputs from
            within :attr:`gnn`. Defaults to the concatenation
            layer.
        num_inducing_points: The number of inducing points for the `VGP`.
            Can only be set for `gp_type='VGP'`.
        kernel: The kernel to use. Defaults to a radial basis function.
        training_stage: The stage of training the model is at.
            Only applies when loading a model.
        sf: The pre-calculated scaling factor. Only applicable when loading
            a pre-trained model.
        **kwargs: Keyword arguments to pass to :meth:`make_gnn`.

    Attributes:
        gnn: The GNN model.
        gp: The GP model.
        train_structs: The training structures.
        train_targets: The training targets.
        val_structs: The validation structures.
        val_targets: The validation targets.
        gp_type: The method to use for the Gaussian process.
            One of 'GP' or 'VGP'.
        save_dir: The directory to save files to during training.
            Files include GNN and GP checkpoints.
        ntarget: The number of target variables.
        layer_index: The index of the layer to extract outputs from
            within :attr:`gnn`.
        num_inducing_points: The number of inducing points for the `VGP`.
            Shoud be `None` for `gp_type='GP'`.
        kernel: The kernel to use. `None` means a radial basis function.
        sf: The scaling factor. Defaults to `None` when uncalculated.
        gnn_ckpt_path: The path to the GNN checkpoints.
        gnn_save_path: The path to the saved GNN.
        gp_ckpt_path: The path to the GP checkpoints.
        gp_save_path: The path to the saved GP.
        kernel_save_path: The path to the saved kernel.
        data_save_path: The path to the saved serialized data needed for
            reloading the GP: see :meth:`_gen_serial_data`.
        train_database: The path to the training database.
        val_database: The path to the validation database.
        sf_path: The path to the saved :attr:`sf`.
        meta_path: The path to the saved metadata: see :meth:`_write_metadata`.

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
        kernel: Optional[tfp.math.psd_kernels.PositiveSemidefiniteKernel] = None,
        training_stage: int = 0,
        sf: Optional[np.ndarray] = None,
        **kwargs,
    ):
        """Initialize class."""
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
        self.sf = sf
        self.layer_index = layer_index
        self.num_inducing_points = num_inducing_points
        self.kernel = kernel

        self.gnn_ckpt_path = self.save_dir / "gnn_ckpts"
        self.gnn_save_path = self.save_dir / "gnn_model"
        self.gp_ckpt_path = self.save_dir / "gp_ckpts"
        self.gp_save_path = self.save_dir / "gp_model"
        self.kernel_save_path = self.save_dir / "kernel.pkl"

        self.data_save_path = self.save_dir / "data"
        self.train_database = self.data_save_path / "train.fthr"
        self.val_database = self.data_save_path / "val.fthr"
        self.sf_path = self.data_save_path / "sf.npy"
        self.meta_path = self.data_save_path / "meta.txt"

        # * Make directories
        for direct in [self.save_dir, self.data_save_path]:
            os.makedirs(direct, exist_ok=True)

        self.gnn: GNN = (
            self.make_gnn(**kwargs) if training_stage == 0 else self.load_gnn()
        )

        # Initialize GP
        if training_stage < 2:
            self.gp: Optional[Union[GPTrainer, SingleLayerVGP]] = None
        else:
            index_points = np.stack(self.get_index_points(self.train_structs))
            index_points = convert_index_points(index_points)

            if gp_type == "VGP":
                # Should already have been caught, but for the type checker's sake
                assert num_inducing_points is not None
                self.gp = SingleLayerVGP(
                    index_points,
                    num_inducing_points,
                    ntarget,
                    prev_model=str(self.gp_save_path),
                    kernel=self.kernel,
                )

            else:
                targets = convert_index_points(np.stack(self.train_targets))
                self.gp = GPTrainer(
                    index_points, targets, self.gp_ckpt_path, self.kernel
                )

            self.kernel = self.gp.kernel

    @abstractmethod
    def make_gnn(self, **kwargs) -> GNN:
        """Construct a new GNN."""
        raise NotImplementedError()

    @abstractmethod
    def load_gnn(self) -> GNN:
        """Load a pre-trained GNN."""
        raise NotImplementedError()

    @property
    def training_stage(self) -> Literal[0, 1, 2]:
        """Indicate the training stage the model is at.

        Returns:
            training_stage: How much of the model is trained.
                Can take one of three values:

                * 0 - Untrained.
                * 1 - :attr:`gnn` trained.
                * 2 - :attr:`gnn` and :attr:`gp` trained.

        """
        return int(self.gnn_save_path.exists()) + bool(self.gp)  # type: ignore

    @abstractmethod
    def train_gnn(self):
        """Train the GNN."""
        pass

    def _update_sf(self):
        """Update the saved scaling factor.

        This must be called to update :attr:`sf` whenever the MEGNetModel
        is updated (i.e. trained).

        """
        ls = LayerScaler.from_train_data(
            self.gnn, self.train_structs, layer_index=self.layer_index
        )
        self.sf = ls.sf

    def get_index_points(
        self, structures: List[pymatgen.Structure]
    ) -> List[np.ndarray]:
        """Determine and preprocess index points for GP training.

        Args:
            structures: A list of structures to convert to inputs.

        Returns:
            index_points: The feature arrays of the structures.

        """
        ls = LayerScaler(self.gnn, self.sf, self.layer_index)
        return ls.structures_to_input(structures)

    def train_uq(
        self, epochs: int = 500, **kwargs
    ) -> Iterator[Optional[Dict[str, float]]]:
        """Train the uncertainty quantifier.

        Extracts chosen layer outputs from :attr:`gnn`,
        scale them and train the appropriate GP (from :attr:`gp_type`).

        Yields:
            metrics: The calculated metrics at every step of training.
                (Only for `gp_type='GP'`).

        """
        training_idxs = np.stack(self.get_index_points(self.train_structs))
        val_idxs = np.stack(self.get_index_points(self.val_structs))

        training_idxs = convert_index_points(training_idxs)
        val_idxs = convert_index_points(val_idxs)

        if self.gp_type == "GP":
            yield from self._train_gp(training_idxs, val_idxs, epochs, **kwargs)
        else:
            self._train_vgp(training_idxs, val_idxs, epochs, **kwargs)
            yield None

    def _train_gp(
        self,
        train_idxs: List[np.ndarray],
        val_idxs: List[np.ndarray],
        epochs: int,
        **kwargs,
    ) -> Iterator[Dict[str, float]]:
        """Train a GP on preprocessed layer outputs from a model."""
        if self.gp_type != "GP":
            raise ValueError("Can only train GP for `gp_type='GP'`")

        train_targets = tf.constant(np.stack(self.train_targets), dtype=tf.float64)
        val_targets = tf.constant(np.stack(self.val_targets), dtype=tf.float64)

        gp_trainer = GPTrainer(
            train_idxs,
            train_targets,
            checkpoint_dir=str(self.gp_ckpt_path),
            kernel=self.kernel,
        )
        yield from gp_trainer.train_model(
            val_idxs, val_targets, epochs, save_dir=str(self.gp_save_path), **kwargs
        )

        self.kernel = self.gp.kernel

    def _train_vgp(
        self,
        train_idxs: List[np.ndarray],
        val_idxs: List[np.ndarray],
        epochs: int,
        **kwargs,
    ) -> None:
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
        self.gp = vgp

        self.kernel = self.gp.kernel

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

        # * Write sf
        if self.sf is not None:
            with self.sf_path.open("wb") as f:
                np.save(f, self.sf)

        # * Write metadata
        self._write_metadata()

        # * Write kernel
        with self.kernel_save_path.open("wb") as f:
            pickle.dump(self.kernel, f)

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

        # ? Currently no need to save preprocessed index_points; the structures suffice
        # if self.training_stage > 0:
        #     data["index_points"] = [
        #         serialize_array(ips) for ips in self.get_index_points(structs)
        #     ]

        return data

    def _write_metadata(self):
        """Write metadata to a file.

        Metadata contains :attr:`gp_type`, :attr:`num_inducing_points`,
        :attr:`layer_index`, :attr:`ntarget` and :attr:`training_stage`,
        as well as the serialised :attr:`sf`.

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
        data["struct"] = data["struct"].apply(pymatgen.Structure.from_str, fmt="json")

        if isinstance(data["target"][0], bytes):
            # Data is serialized
            data["target"] = data["target"].apply(deserialize_array)

        # ? index_points not currently saved
        # try:
        #     data["index_points"] = data["index_points"].apply(deserialize_array)
        # except KeyError:
        #     # No index points in dataset
        #     pass

        return data

    @classmethod
    def load(cls, dirname: Union[Path, str]) -> ProbGNN:
        """Load a full-stack model."""
        data_dir = Path(dirname) / "data"
        train_datafile = data_dir / "train.fthr"
        val_datafile = data_dir / "val.fthr"
        kernel_save_path = Path(dirname) / "kernel.pkl"

        # * Load serialized training + validation data
        train_data = cls._load_serial_data(train_datafile)
        val_data = cls._load_serial_data(val_datafile)

        # * Load metadata
        metafile = data_dir / "meta.txt"
        with metafile.open("r") as f:
            meta = json.load(f)

        # * Load scaling factor, if already calculated
        sf_dir = data_dir / "sf.npy"
        sf = None
        if meta["training_stage"] > 0:
            with sf_dir.open("rb") as f:  # type: ignore
                sf = np.load(f)

        # * Load kernel
        with kernel_save_path.open("rb") as f:
            kernel = pickle.load(f)

        return cls(
            train_data["struct"],
            train_data["target"],
            val_data["struct"],
            val_data["target"],
            save_dir=dirname,
            sf=sf,
            kernel=kernel,
            **meta,
        )


class MEGNetProbModel(ProbGNN):
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
            within :attr:`gnn`. Defaults to the concatenation
            layer.
        num_inducing_points: The number of inducing points for the `VGP`.
            Can only be set for `gp_type='VGP'`.
        training_stage: The stage of training the model is at.
            Only applies when loading a model.
        sf: The pre-calculated scaling factor. Only applicable when loading
            a pre-trained model.
        **kwargs: Keyword arguments to pass to :class:`MEGNetModel`.

    """

    def make_gnn(self, **kwargs) -> MEGNetModel:
        """Create a new MEGNetModel."""
        try:
            meg_model = MEGNetModel(ntarget=self.ntarget, **kwargs)
        except ValueError:
            meg_model = MEGNetModel(
                ntarget=self.ntarget, **kwargs, **get_default_megnet_args()
            )
        return meg_model

    def load_gnn(self) -> MEGNetModel:
        """Load a saved MEGNetModel."""
        return MEGNetModel.from_file(str(self.gnn_save_path))

    def train_gnn(
        self,
        epochs: Optional[int] = 1000,
        batch_size: Optional[int] = 128,
        **kwargs,
    ):
        """Train the MEGNetModel.

        Args:
            epochs: The number of training epochs.
            batch_size: The batch size.
            **kwargs: Keyword arguments to pass to :func:`MEGNetModel.train`.

        """
        self.gnn.train(
            self.train_structs,
            self.train_targets,
            self.val_structs,
            self.val_targets,
            epochs=epochs,
            batch_size=batch_size,
            dirname=self.gnn_ckpt_path,
            **kwargs,
        )

        self.gnn.save_model(str(self.gnn_save_path))
        self._update_sf()


def targets_to_tensor(targets: List[Union[float, np.ndarray]]) -> tf.Tensor:
    """Convert a list of target values to a Tensor."""
    return tf.constant(np.stack(targets), dtype=tf.float64)


def get_default_megnet_args(
    nfeat_bond: int = 10, r_cutoff: float = 5.0, gaussian_width: float = 0.5
) -> dict:
    """Get default MEGNet arguments.

    These are the fallback for when no graph converter is supplied,
    taken from the MEGNet Github page.

    Args:
        nfeat_bond: Number of bond features. Default (10) is very low, useful for testing.
        r_cutoff: The atomic radius cutoff, above which to ignore bonds.
        gaussian_width: The width of the gaussian to use in determining bond features.

    Returns:
        megnet_args: Some default-ish MEGNet arguments.

    """
    gaussian_centers = np.linspace(0, r_cutoff + 1, nfeat_bond)
    graph_converter = CrystalGraph(cutoff=r_cutoff)
    return {
        "graph_converter": graph_converter,
        "centers": gaussian_centers,
        "width": gaussian_width,
    }
