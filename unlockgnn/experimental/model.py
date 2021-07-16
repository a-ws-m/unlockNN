"""Experimental full-stack MEGNetProbModel code."""
import json
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Literal, Optional, Union

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
from megnet.models.megnet import MEGNetModel
from unlockgnn.gp.kernel_layers import KernelLayer, RBFKernelFn
from unlockgnn.gp.vgp_trainer import VariationalLoss


def make_probabilistic(
    gnn: keras.Model,
    num_inducing_points: int,
    kernel: KernelLayer = RBFKernelFn(),
    latent_layer: Union[str, int] = -2,
    ntargets: int = 1,
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
    vgp_outputs = tfp.layers.VariationalGaussianProcess(
        num_inducing_points,
        kernel,
        event_shape=(ntargets,),
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
        ntargets: int = 1,
        metrics: List[Union[str, tf.keras.metrics.Metric]] = ["mae"],
        kl_weight: float = 1.0,
        optimizer: keras.optimizers.Optimizer = tf.optimizers.Adam(),
    ) -> None:
        """Initialize probabilistic model."""
        self.save_path = save_path
        self.weights_path = save_path / "weights.h5"
        self.conf_path = save_path / "config.json"
        # TODO: pre-existing model check and load procedure
        # Save GNN for use in reloading model from disk
        gnn.save(save_path / "gnn", include_optimizer=False)

        # Instantiate probabilistic model
        self.model = make_probabilistic(
            gnn, num_inducing_points, kernel, latent_layer, ntargets
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

    @abstractmethod
    def train_vgp(self):
        """Train the VGP."""
        if not self.gnn_frozen:
            warnings.warn("GNN layers not frozen during VGP training.", RuntimeWarning)

    @abstractmethod
    def train_gnn(self):
        """Train the GNN."""
        if not self.vgp_frozen:
            warnings.warn("VGP layer not frozen during GNN training.", RuntimeWarning)

    @abstractmethod
    def train_full(self):
        """Train the full-stack model."""
        ...

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
    def config(self) -> dict:
        """Get the configuration parameters needed to save to disk."""
        return {var_name: getattr(self, var_name) for var_name in self.CONFIG_VARS}

    def save(self):
        """Save the model to disk."""
        self.model.save_weights(self.weights_path)
        with self.conf_path.open("w") as f:
            json.dump(self.config, f)

    def save_kernel(self):
        """Save the VGP's kernel to disk."""
