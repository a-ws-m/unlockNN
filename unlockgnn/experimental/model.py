"""Experimental full-stack MEGNetProbModel code."""
import json
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union
from megnet.data.graph import GraphBatchDistanceConvert, GraphBatchGenerator
from megnet.utils.preprocessing import DummyScaler

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
from megnet.models import GraphModel
from pymatgen import Structure
from unlockgnn.gp.kernel_layers import KernelLayer, RBFKernelFn
from unlockgnn.gp.vgp_trainer import VariationalLoss

MEGNetGraph = Dict[str, Union[np.ndarray, List[Union[int, float]]]]
Targets = List[Union[float, np.ndarray]]


def make_probabilistic(
    gnn: keras.Model,
    num_inducing_points: int,
    kernel: KernelLayer = RBFKernelFn(),
    latent_layer: Union[str, int] = -2,
    target_shape: Union[Sequence, int] = 1,
) -> keras.Model:
    """Make a GNN probabilistic by replacing the final layer(s) with a VGP.

    Caution: This function modifies the GNN in memory. Ensure that the GNN has
    been saved to disk before using.

    Args: gnn: The base GNN model to modify. latent_layer: The name or index of
        the layer of the GNN to be fed into the VGP.

    """
    # Determine how many layers to pop
    if isinstance(latent_layer, int):
        latent_idx = latent_layer
    else:
        latent_idx = [layer.name for layer in gnn.layers].index(latent_layer)

    if latent_idx > 0:
        num_pops = len(gnn.layers) - latent_idx - 1
    else:
        num_pops = -latent_idx - 1

    # Remove layers up to the specified one
    for _ in range(num_pops):
        gnn.layers.pop()

    inputs = gnn.layers[0]
    vgp_input = gnn(inputs)
    output_shape = (
        (target_shape,) if isinstance(target_shape, int) else tuple(target_shape)
    )
    vgp_outputs = tfp.layers.VariationalGaussianProcess(
        num_inducing_points,
        kernel,
        event_shape=output_shape,
        jitter=1e-06,
        convert_to_tensor_fn=tfp.distributions.Distribution.mean,
    )(vgp_input)

    return keras.Model(inputs, vgp_outputs)


class ProbGNN(ABC):
    """Wrapper for creating a probabilistic GNN model.

    Args:
        gnn: The base GNN model to modify.
        optimizer: The model optimizer, needed for recompilation.
        latent_layer: The name or index of the layer of the GNN to be fed into
            the VGP.

    """

    CONFIG_VARS: List[str] = [
        "num_inducing_points",
        "ntargets",
        "metrics",
        "kl_weight",
    ]

    def __init__(
        self,
        gnn: keras.Model,
        num_inducing_points: int,
        save_path: Path,
        kernel: KernelLayer = RBFKernelFn(),
        latent_layer: Union[str, int] = -2,
        target_shape: Union[Sequence, int] = 1,
        metrics: List[Union[str, tf.keras.metrics.Metric]] = ["mae"],
        kl_weight: float = 1.0,
        optimizer: keras.optimizers.Optimizer = tf.optimizers.Adam(),
    ) -> None:
        """Initialize probabilistic model."""
        self.save_path = save_path
        self.weights_path = save_path / "weights.h5"
        self.ckpt_path = save_path / "checkpoint.h5"
        self.conf_path = save_path / "config.json"
        self.kernel_path = save_path / "kernel"

        self.kernel = kernel

        # TODO: pre-existing model check and load procedure
        # Save GNN for use in reloading model from disk
        gnn.save(save_path / "gnn", include_optimizer=False)

        # Instantiate probabilistic model
        self.model = make_probabilistic(
            gnn, num_inducing_points, self.kernel, latent_layer, target_shape
        )

        self.metrics = metrics
        # Freeze GNN layers and compile, ready to train the VGP
        self.set_frozen("GNN", kl_weight=kl_weight, optimizer=optimizer)

    def set_frozen(
        self,
        layers: Literal["GNN", "VGP"],
        freeze: bool = True,
        recompile: bool = True,
        **compilation_kwargs,
    ):
        """Freeze or thaw probabilistic GNN layers."""
        if layers == "GNN":
            for layer in self.model.layers[:-1]:
                layer.trainable = not freeze
        elif layers == "VGP":
            self.model.layers[-1].trainable = not freeze
        else:
            raise ValueError(f"Expected one of 'GNN' or 'VGP', got {layers=}")

        if recompile:
            self.compile(**compilation_kwargs)

    def train_vgp(self, *args, **kwargs):
        """Train the VGP."""
        if not self.gnn_frozen:
            warnings.warn("GNN layers not frozen during VGP training.", RuntimeWarning)
        self.train(*args, **kwargs)

    def train_gnn(self, *args, **kwargs):
        """Train the GNN."""
        if not self.vgp_frozen:
            warnings.warn("VGP layer not frozen during GNN training.", RuntimeWarning)
        self.train(*args, **kwargs)

    @abstractmethod
    def train(self, *args, **kwargs):
        """Train the full-stack model."""
        ...

    def predict(self, input):
        """Predict target values and uncertainties for a given input."""
        return self.model(input)

    @property
    def gnn_frozen(self) -> bool:
        """Determine whether all GNN layers are frozen."""
        return all((not layer.trainable for layer in self.model.layers[:-1]))

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
    def config(self) -> dict:
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


class MEGNetProbModel(ProbGNN):
    """ProbGNN for MEGNetModels."""

    def __init__(
        self,
        meg_model: GraphModel,
        num_inducing_points: int,
        save_path: Path,
        kernel: KernelLayer = RBFKernelFn(),
        latent_layer: Union[str, int] = -2,
        target_shape: Optional[int] = None,
        metrics: List[Union[str, tf.keras.metrics.Metric]] = ["mae"],
        kl_weight: float = 1.0,
        optimizer: keras.optimizers.Optimizer = tf.optimizers.Adam(),
    ) -> None:
        """Initialize probabilistic model."""
        if target_shape is None:
            # Determine output shape based on MEGNetModel
            target_shape = meg_model.model.layers[-1].output_shape
        self.meg_model = meg_model

        super().__init__(
            self.meg_model.model,
            num_inducing_points,
            save_path,
            kernel,
            latent_layer,
            target_shape,
            metrics,
            kl_weight,
            optimizer,
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
        """Train the model."""
        # Convert structures to graphs for model input
        train_gen, train_graphs = self.create_input_generator(
            structs, targets, batch_size, scrub_failed_structs
        )
        val_gen = None
        val_graphs = None
        if val_structs and val_targets:
            val_gen, val_graphs = self.create_input_generator(
                val_structs, val_targets, batch_size, scrub_failed_structs
            )

        # Configure callbacks
        if use_default_ckpt_handler:
            callbacks.append(self.ckpt_callback)

        # Train
        steps_per_train = int(np.ceil(len(train_graphs) / batch_size))
        steps_per_val = int(np.ceil(len(val_graphs) / batch_size))

        self.model.fit(
            train_gen,
            steps_per_epoch=steps_per_train,
            validation_data=val_gen,
            validation_steps=steps_per_val,
            epochs=epochs,
            verbose=verbose,
            callbacks=callbacks,
        )

    def scale_targets(self, targets: Targets, num_atoms: List[int]) -> Targets:
        """Scale target values using underlying MEGNetModel's scaler."""
        return [
            self.meg_model.target_scaler.transform(target, num_atom)
            for target, num_atom in zip(targets, num_atoms)
        ]

    def create_input_generator(
        self,
        structs: List[Structure],
        targets: Targets,
        batch_size: int,
        scrub_failed_structs: bool = False,
    ) -> Tuple[
        Union[GraphBatchDistanceConvert, GraphBatchGenerator], List[MEGNetGraph]
    ]:
        """Create generator for use during training and validation of model."""
        graphs, trunc_targets = self.meg_model.get_all_graphs_targets(
            structs, targets, scrub_failed_structs
        )
        # Check dimensions of model against converted graphs
        self.meg_model.check_dimension(graphs[0])

        # Scale targets if necessary
        if not isinstance(self.meg_model.target_scaler, DummyScaler):
            num_atoms = [len(graph["atom"]) for graph in graphs]
            trunc_targets = self.scale_targets(trunc_targets, num_atoms)

        inputs = self.meg_model.graph_converter.get_flat_data(graphs, trunc_targets)
        return self.meg_model._create_generator(*inputs, batch_size=batch_size), graphs
