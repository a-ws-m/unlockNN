"""Experimental full-stack MEGNetProbModel code."""
import json
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

try:
    from typing import Literal, get_args
except ImportError:
    from typish import Literal, get_args


import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
from megnet.data.graph import GraphBatchDistanceConvert, GraphBatchGenerator
from megnet.models import MEGNetModel
from megnet.utils.preprocessing import DummyScaler
from pymatgen.core import Structure
from tensorflow.python.keras.utils import losses_utils

from .kernel_layers import KernelLayer, RBFKernelFn, load_kernel

tfd = tfp.distributions

MEGNetGraph = Dict[str, Union[np.ndarray, List[Union[int, float]]]]
Targets = List[Union[float, np.ndarray]]
LayerName = Literal["GNN", "VGP"]

__all__ = ["ProbGNN", "MEGNetProbModel"]


class VariationalLoss(keras.losses.Loss):
    """Implementation of variational loss using keras API."""

    def __init__(
        self,
        kl_weight: float,
        reduction: keras.losses.Reduction = losses_utils.ReductionV2.AUTO,
        name: str = "variational_loss",
    ):
        """Initialize loss function and KL divergence loss scaling factor."""
        self.kl_weight = kl_weight
        super().__init__(reduction=reduction, name=name)

    def call(self, y_true, predicted_distribution):
        """Calculate the variational loss."""
        return predicted_distribution.variational_loss(y_true, kl_weight=self.kl_weight)


def make_probabilistic(
    gnn: keras.Model,
    num_inducing_points: int,
    kernel: KernelLayer = RBFKernelFn(),
    latent_layer: Union[str, int] = -2,
    target_shape: Union[Tuple[int], int] = 1,
    prediction_mode: bool = False,
) -> keras.Model:
    """Make a GNN probabilistic by replacing the final layer(s) with a VGP.

    Caution: This function modifies the GNN in memory. Ensure that the GNN has
    been saved to disk before using.

    Args:
        gnn: The base GNN model to modify. latent_layer: The name or index of
            the layer of the GNN to be fed into the VGP.
        num_inducing_points: The number of inducing index points for the
            VGP.
        kernel: A :class`KernelLayer` for the VGP to use.
        latent_layer: The index or name of the GNN layer to use as the
            input for the VGP.
        target_shape: The shape of the target values.
        prediction_mode: Whether to create a model for predictions _only_.
            (Resulting model cannot be serialized and loss functions won't work.)

    Returns:
        A `keras.Model` with the `gnn`'s first layers, but terminating in a
            VGP.

    """
    # Determine how many layers to pop
    if isinstance(latent_layer, int):
        latent_idx = latent_layer
    else:
        latent_idx = [layer.name for layer in gnn.layers].index(latent_layer)

    vgp_input = gnn.layers[latent_idx].output

    output_shape = (
        (target_shape,) if isinstance(target_shape, int) else tuple(target_shape)
    )
    convert_fn = (
        tfd.Distribution.mean
        if not prediction_mode
        else lambda trans_dist: tf.concat(
            [trans_dist.distribution.mean(), trans_dist.distribution.stddev()], axis=-1
        )
    )
    vgp_outputs = tfp.layers.VariationalGaussianProcess(
        num_inducing_points,
        kernel,
        event_shape=output_shape,
        inducing_index_points_initializer=None,  # TODO: Clever clustering initialization
        convert_to_tensor_fn=convert_fn,
    )(vgp_input)

    return keras.Model(gnn.inputs, vgp_outputs)


class ProbGNN(ABC):
    """Wrapper for creating a probabilistic GNN model.

    Args:
        num_inducing_points: The number of inducing index points for the
            VGP.
        save_path: Path to the save directory for the model.
        gnn: The base GNN model to modify.
        kernel: A :class`KernelLayer` for the VGP to use.
        latent_layer: The name or index of the layer of the GNN to be fed into
            the VGP.
        target_shape: The shape of the target values.
        metrics: A list of metrics to record during training.
        kl_weight: The relative weighting of the Kullback-Leibler divergence
            in the loss function.
        optimizer: The model optimizer, needed for recompilation.
        load_ckpt: Whether to load the best checkpoint's weights, instead
            of those saved at the time of the last :meth:`save`.

    """

    CONFIG_VARS: List[str] = [
        "num_inducing_points",
        "metrics",
        "kl_weight",
        "latent_layer",
        "target_shape",
    ]

    def __init__(
        self,
        num_inducing_points: int,
        save_path: Path,
        gnn: Optional[keras.Model] = None,
        kernel: KernelLayer = RBFKernelFn(),
        latent_layer: Union[str, int] = -2,
        target_shape: Union[Tuple[int], int] = 1,
        metrics: List[Union[str, tf.keras.metrics.Metric]] = ["mae"],
        kl_weight: float = 1.0,
        optimizer: keras.optimizers.Optimizer = tf.optimizers.Adam(),
        load_ckpt: bool = True,
    ) -> None:
        """Initialize the probabilistic model.

        Saves the GNN to disk, loads weights from disk if they exist and then
        instantiates the probabilistic model. The model's GNN layers are
        initially frozen by default (but not the VGP).

        """
        self.save_path = save_path
        self.weights_path = save_path / "weights"
        self.ckpt_path = save_path / "checkpoint.h5"
        self.conf_path = save_path / "config.json"
        self.kernel_path = save_path / "kernel"
        self.gnn_path = save_path / "gnn"

        self.kernel = kernel
        self.metrics = metrics
        self.optimizer: keras.optimizers.Optimizer = optimizer
        self.kl_weight = kl_weight
        self.latent_layer = latent_layer
        self.target_shape = target_shape
        self.num_inducing_points = num_inducing_points

        self.pred_model: Optional[keras.Model] = None

        if not self.gnn_path.exists():
            loading: bool = False
            if gnn is None:
                raise IOError(
                    f"{self.gnn_path} does not exist."
                    " Please check the `save_path`, or pass a GNN model if creating a new `ProbGNN`."
                )
            # Save GNN for use in reloading model from disk
            gnn.save(self.gnn_path, include_optimizer=False)
        else:
            # We're loading from memory
            loading = True
            gnn = keras.models.load_model(self.gnn_path, compile=False)

            try:
                self.kernel = load_kernel(self.kernel_path)
            except FileNotFoundError:
                warnings.warn("No saved kernel found.")

        # Instantiate probabilistic model
        self.model = make_probabilistic(
            gnn,
            num_inducing_points,
            self.kernel,
            latent_layer,
            target_shape,
        )

        # Freeze GNN layers and compile, ready to train the VGP
        self.set_frozen("GNN")

        if loading:
            # Load weights from the relevant source
            to_load = self.ckpt_path if load_ckpt else self.weights_path
            try:
                self.model.load_weights(to_load)
            except OSError:
                warnings.warn(f"No saved weights found at `{to_load}`.")

    def set_frozen(
        self,
        layers: Union[LayerName, List[LayerName]],
        freeze: bool = True,
        recompile: bool = True,
        **compilation_kwargs,
    ) -> None:
        """Freeze or thaw probabilistic GNN layers.

        Args:
            layers: Name or list of names of layers to thaw.
            freeze: Whether to freeze (`True`) or thaw (`False`) the layers.
            recompile: Whether to recompile the model after the operation.
            **compilation_kwargs: Keyword arguments to pass to :meth:`compile`.

        Raises:
            ValueError: If one or more `layers` are invalid names.

        """
        if not isinstance(layers, list):
            layers = [layers]

        # Validation check
        valid_layers = get_args(LayerName)
        for name in layers:
            if name not in valid_layers:
                raise ValueError(
                    f"Cannot freeze `{name}`; must be one of {valid_layers}."
                )

        if "GNN" in layers:
            last_gnn_index = -1
            for layer in self.model.layers[:last_gnn_index]:
                layer.trainable = not freeze
        if "VGP" in layers:
            self.model.layers[-1].trainable = not freeze

        if recompile:
            self.compile(
                kl_weight=self.kl_weight, optimizer=self.optimizer, **compilation_kwargs
            )

    @abstractmethod
    def train(self, *args, **kwargs) -> None:
        """Train the full-stack model."""
        ...

    def update_pred_model(self, force_new: bool = False) -> None:
        """Instantiate or update the predictor model.

        The predictor model is saved in :attr:`pred_model`. This method is a
        workaround to reconcile the inability to save or train a model that
        returns the VGP distribution's mean _and_ standard deviation
        simultaneously.

        This method creates a clone model and so it must be called before making
        a prediction whenever the :attr:`model`'s weights have changed. By
        default, the method checks whether the pre-existing :attr:`pred_model`'s
        weights are similar to the :attr:`model`'s weights before cloning, and
        skips execution if they are. Setting `force_new=True` skips this check.

        Args:
            force_new: Whether to force the creation of a new model, skipping
                the weights equality check.

        """
        while not force_new:
            # Check if we need to instantiate the prediction model
            if self.pred_model is None:
                force_new = True
                break
            current_weights = self.model.get_weights()
            predictor_weights = self.pred_model.get_weights()
            force_new = not all(
                np.allclose(current, predictor, equal_nan=True)
                for current, predictor in zip(current_weights, predictor_weights)
            )
            break
        if force_new:
            self.model.save_weights(self.weights_path)
            gnn = keras.models.load_model(self.gnn_path, compile=False)
            pred_model = make_probabilistic(
                gnn,
                self.num_inducing_points,
                self.kernel,
                self.latent_layer,
                self.target_shape,
                prediction_mode=True,
            )
            pred_model.compile()
            pred_model.load_weights(self.weights_path)
            self.pred_model = pred_model

    def predict(self, input) -> np.ndarray:
        """Predict target values and standard deviations for a given input.

        Args:
            input: The input(s) to the model.

        Returns:
            predictions: A numpy array containing predicted means and
                standard deviations.

        """
        self.update_pred_model()
        return self.pred_model.predict(input)

    @property
    def gnn_frozen(self) -> bool:
        """Determine whether all GNN layers are frozen."""
        last_gnn_index = -1
        return all(
            (not layer.trainable for layer in self.model.layers[:last_gnn_index])
        )

    @property
    def vgp_frozen(self) -> bool:
        """Determine whether the VGP is frozen."""
        return not self.model.layers[-1].trainable

    def compile(
        self,
        kl_weight: float = 1.0,
        optimizer: keras.optimizers.Optimizer = tf.optimizers.Adam(),
    ):
        """Compile the probabilistic GNN.

        Recompilation is required whenever layers are (un)frozen.

        Args:
            kl_weight: The relative weighting of the Kullback-Leibler divergence
                in the loss function.
            optimizer: The model optimizer, needed for recompilation.

        """
        self.model.compile(
            optimizer, loss=VariationalLoss(kl_weight), metrics=self.metrics
        )

    @property
    def ckpt_callback(self):
        """Get the default configuration for a model checkpoint callback."""
        return tf.keras.callbacks.ModelCheckpoint(
            self.ckpt_path, save_best_only=True, save_weights_only=True
        )

    @property
    def config(self) -> Dict[str, Any]:
        """Get the configuration parameters needed to save to disk."""
        return {var_name: getattr(self, var_name) for var_name in self.CONFIG_VARS}

    def save(self):
        """Save the model to disk."""
        self.model.save_weights(self.weights_path)
        with self.conf_path.open("w") as f:
            json.dump(self.config, f)
        self.save_kernel()

    def save_kernel(self):
        """Save the VGP's kernel to disk."""
        self.kernel.save(self.kernel_path)

    @classmethod
    def load(cls: "ProbGNN", save_path: Path, load_ckpt: bool = True) -> "ProbGNN":
        """Load a ProbGNN from disk.

        Args:
            save_path: The path to the model's save directory.
            load_ckpt: Whether to load the best checkpoint's weights, instead
                of those saved at the time of the last :meth:`save`.

        Returns:
            The loaded model.

        Raises:
            IOError: If the `save_path` does not exist.

        """
        # Check the save path exists
        if not save_path.exists():
            raise IOError(f"{save_path} does not exist.")

        config_path = save_path / "config.json"
        with config_path.open("r") as f:
            config = json.load(f)

        return cls(save_path=save_path, load_ckpt=load_ckpt, **config)


class MEGNetProbModel(ProbGNN):
    """ProbGNN for MEGNetModels.

    Args:
        num_inducing_points: The number of inducing index points for the
            VGP.
        save_path: Path to the save directory for the model.
        meg_model: The base `MEGNetModel` to modify.
        kernel: A :class`KernelLayer` for the VGP to use.
        latent_layer: The name or index of the layer of the GNN to be fed into
            the VGP.
        target_shape: The shape of the target values.
        metrics: A list of metrics to record during training.
        kl_weight: The relative weighting of the Kullback-Leibler divergence
            in the loss function.
        optimizer: The model optimizer, needed for recompilation.
        load_ckpt: Whether to load the best checkpoint's weights, instead
            of those saved at the time of the last :meth:`save`.

    """

    def __init__(
        self,
        num_inducing_points: int,
        save_path: Path,
        meg_model: Optional[MEGNetModel] = None,
        kernel: KernelLayer = RBFKernelFn(),
        latent_layer: Union[str, int] = -2,
        target_shape: Optional[Union[Tuple[int], int]] = None,
        metrics: List[Union[str, tf.keras.metrics.Metric]] = ["mae"],
        kl_weight: float = 1.0,
        optimizer: keras.optimizers.Optimizer = tf.optimizers.Adam(),
        load_ckpt: bool = True,
    ) -> None:
        """Initialize probabilistic model."""
        self.meg_save_path = save_path / "megnet"
        if self.meg_save_path.exists():
            # Load from memory
            self.meg_model = MEGNetModel.from_file(str(self.meg_save_path))
        else:
            if meg_model is None:
                raise IOError(
                    f"{self.meg_save_path} does not exist."
                    " Please check the `save_path` or pass a `meg_model` if creating a new `MEGNetProbModel`."
                )
            self.meg_model = meg_model
            self.meg_model.save_model(str(self.meg_save_path))

        if target_shape is None:
            # Determine output shape based on MEGNetModel
            target_shape = self.meg_model.model.layers[-1].output_shape
            target_shape = tuple(dim for dim in target_shape if dim is not None)

        super().__init__(
            num_inducing_points,
            save_path,
            self.meg_model.model,
            kernel,
            latent_layer,
            target_shape,
            metrics,
            kl_weight,
            optimizer,
            load_ckpt,
        )

    def train(
        self,
        structs: List[Structure],
        targets: Targets,
        epochs: int,
        val_structs: Optional[List[Structure]] = None,
        val_targets: Optional[Targets] = None,
        callbacks: List[tf.keras.callbacks.Callback] = [],
        use_default_ckpt_handler: bool = True,
        batch_size: int = 128,
        scrub_failed_structs: bool = False,
        verbose: Literal[0, 1, 2] = 2,
    ):
        """Train the model.

        Args:
            structs: A list of training crystal structures.
            targets: A list of training target values.
            epochs: The number of training epochs.
            val_structs: A list of validation crystal structures.
            val_targets: A list of validation target values.
            callbacks: A list of additional callbacks.
            use_default_ckpt_handler: Whether to use the default
                checkpoint callback.
            batch_size: The batch size for training and validation.
            scrub_failed_structures: Whether to discard structures
                that could not be converted to graphs.
            verbose: The level of verbosity. See Keras's documentation
                on `Model.fit`.

        """
        # Convert structures to graphs for model input
        train_gen, train_graphs = self.create_input_generator(
            structs, targets, batch_size, scrub_failed_structs
        )
        steps_per_train = int(np.ceil(len(train_graphs) / batch_size))

        val_gen = None
        val_graphs = None
        steps_per_val = None
        if val_structs is not None and val_targets is not None:
            val_gen, val_graphs = self.create_input_generator(
                val_structs, val_targets, batch_size, scrub_failed_structs
            )
            steps_per_val = int(np.ceil(len(val_graphs) / batch_size))

        # Configure callbacks
        if use_default_ckpt_handler:
            callbacks.append(self.ckpt_callback)

        # Train
        self.model.fit(
            train_gen,
            steps_per_epoch=steps_per_train,
            validation_data=val_gen,
            validation_steps=steps_per_val,
            epochs=epochs,
            verbose=verbose,
            callbacks=callbacks,
        )

    def evaluate(
        self,
        eval_structs: List[Structure],
        eval_targets: Targets,
        batch_size: int = 128,
        scrub_failed_structs: bool = False,
    ) -> Dict[str, float]:
        """Evaluate model metrics.

        Args:
            eval_structs: Structures on which to evaluate performance.
            eval_targets: True target values for structures.
            batch_size: The batch size for training and validation.
            scrub_failed_structures: Whether to discard structures
                that could not be converted to graphs.

        Returns:
            Dictionary of {metric: value}.

        """
        eval_gen, eval_graphs = self.create_input_generator(
            eval_structs, eval_targets, batch_size, scrub_failed_structs
        )
        steps = int(np.ceil(len(eval_graphs) / batch_size))
        return self.model.evaluate(
            eval_gen, batch_size=batch_size, steps=steps, return_dict=True
        )

    def scale_targets(self, targets: Targets, num_atoms: List[int]) -> Targets:
        """Scale target values using underlying MEGNetModel's scaler.

        Args:
            targets: A list of target values.
            num_atoms: A list of the number of atoms in each structure
                corresponding to the target values.

        Returns:
            The scaled target values.

        """
        return [
            self.meg_model.target_scaler.transform(target, num_atom)
            for target, num_atom in zip(targets, num_atoms)
        ]

    def predict(
        self, input: Union[Structure, Iterable[Structure]], batch_size: int = 128
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict target values and standard deviations for a given input.

        Args:
            input: The input structure(s).
            batch_size: The batch size for predictions.

        Returns:
            means: The mean values of the predicted distribution(s).
            stddevs: The standard deviations of the predicted distribution(s).

        """
        to_freeze = ["GNN", "VGP"]
        self.set_frozen(to_freeze)

        if isinstance(input, Structure):
            # Just one to predict
            input = [input]

        n_inputs = len(input)
        inputs, graphs = self.create_input_generator(
            structs=input, batch_size=batch_size, shuffle=False
        )
        num_atoms = [len(graph["atom"]) for graph in graphs]

        prediction = super().predict(inputs).squeeze()
        means, stddevs = prediction[:n_inputs], prediction[n_inputs:]

        if not isinstance(self.meg_model.target_scaler, DummyScaler):
            means = self.meg_model.target_scaler.inverse_transform(means, num_atoms)
            stddevs = self.meg_model.target_scaler.inverse_transform(stddevs, num_atoms)

        return means, stddevs

    def create_input_generator(
        self,
        structs: List[Structure],
        targets: Optional[Targets] = None,
        batch_size: int = 128,
        scrub_failed_structs: bool = False,
        shuffle: bool = True,
    ) -> Tuple[
        Union[GraphBatchDistanceConvert, GraphBatchGenerator], List[MEGNetGraph]
    ]:
        """Create generator for model inputs.

        Args:
            structs: The input structures.
            targets: The input targets, if any.
            batch_size: The batch size for the generator.
            scrub_failed_structures: Whether to discard structures
                that could not be converted to graphs.
            shuffle: Whether the generator should shuffle the order of the
                structure/target pairs.

        Returns:
            input_generator: The input generator
            graphs: A list of the model input graphs.

        """
        # Make some targets up for compatibility
        has_targets = targets is not None
        target_buffer = targets if has_targets else [0.0] * len(structs)

        graphs, trunc_targets = self.meg_model.get_all_graphs_targets(
            structs, target_buffer, scrub_failed_structs
        )
        # Check dimensions of model against converted graphs
        self.meg_model.check_dimension(graphs[0])

        # Scale targets if necessary
        if not isinstance(self.meg_model.target_scaler, DummyScaler) and has_targets:
            num_atoms = [len(graph["atom"]) for graph in graphs]
            trunc_targets = self.scale_targets(trunc_targets, num_atoms)

        inputs = self.meg_model.graph_converter.get_flat_data(graphs, trunc_targets)
        return (
            self.meg_model._create_generator(
                *inputs, batch_size=batch_size, is_shuffle=shuffle
            ),
            graphs,
        )
