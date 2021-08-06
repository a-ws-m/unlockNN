"""Kernel layer implementations for use with VariationalGaussianProcesses."""
from abc import ABC, abstractmethod
import json
from os import mkdir
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

__all__ = [
    "KernelLayer",
    "AmpAndLengthScaleFn",
    "RBFKernelFn",
    "MaternOneHalfFn",
    "load_kernel",
]


def config_path(directory: Path) -> Path:
    return directory / "config.json"


def params_path(directory: Path) -> Path:
    return directory / "params.npy"


class KernelLayer(tf.keras.layers.Layer, ABC):
    """An ABC for kernel function implementations that work with keras."""

    def call(self, x):
        """Do nothing -- a placeholder for keras."""
        # Never called -- this is just a layer so it can hold variables
        # in a way Keras understands.
        return x

    @property
    @abstractmethod
    def kernel(self) -> tfp.math.psd_kernels.PositiveSemidefiniteKernel:
        """Get a callable kernel."""
        pass

    @property
    @abstractmethod
    def config(self) -> dict:
        """Get configuration information for the class."""
        ...

    def save(self, directory: Path):
        """Save the kernel to disk.

        Subclasses should update this method to save configuration information
        as well. See also :func:`load_kernel`.

        """
        if not directory.exists():
            mkdir(directory)
        weights = self.get_weights()
        np.save(params_path(directory), weights, allow_pickle=True)

        with config_path(directory).open("w") as f:
            json.dump(self.config, f)


class AmpAndLengthScaleFn(KernelLayer, ABC):
    """An ABC for kernels with amplitude and length scale parameters.

    Attributes:
        _amplitude (tf.Tensor): The amplitude of the kernel.
        _length_scale (tf.Tensor): The length scale of the kernel.

    """

    def __init__(self, **kwargs):
        """Initialize layer and parameters."""
        super().__init__(**kwargs)
        dtype = kwargs.get("dtype", tf.float64)

        self._amplitude = self.add_weight(
            initializer=tf.constant_initializer(0), dtype=dtype, name="amplitude"
        )

        self._length_scale = self.add_weight(
            initializer=tf.constant_initializer(0), dtype=dtype, name="length_scale"
        )


class RBFKernelFn(AmpAndLengthScaleFn):
    """A radial basis function implementation that works with keras.

    Attributes:
        _amplitude (tf.Tensor): The amplitude of the kernel.
        _length_scale (tf.Tensor): The length scale of the kernel.

    """

    @property
    def kernel(self) -> tfp.math.psd_kernels.PositiveSemidefiniteKernel:
        """Get a callable kernel."""
        return tfp.math.psd_kernels.ExponentiatedQuadratic(
            amplitude=tf.nn.softplus(0.1 * self._amplitude),
            length_scale=tf.nn.softplus(5.0 * self._length_scale),
        )

    @property
    def config(self) -> dict:
        return {"type": "rbf"}


class MaternOneHalfFn(AmpAndLengthScaleFn):
    """A Matern kernel with parameter 1/2 implementation that works with keras.

    Attributes:
        _amplitude (tf.Tensor): The amplitude of the kernel.
        _length_scale (tf.Tensor): The length scale of the kernel.

    """

    @property
    def kernel(self) -> tfp.math.psd_kernels.PositiveSemidefiniteKernel:
        """Get a callable kernel."""
        return tfp.math.psd_kernels.MaternOneHalf(
            amplitude=tf.nn.softplus(0.1 * self._amplitude),
            length_scale=tf.nn.softplus(5.0 * self._length_scale),
        )

    @property
    def config(self) -> dict:
        return {"type": "matern"}


KERNEL_TYPES = {"rbf": RBFKernelFn, "matern": MaternOneHalfFn}


def load_kernel(directory: Path, kernel_type: Optional[Callable] = None) -> KernelLayer:
    """Load a kernel from the disk.

    Args:
        directory: Path to the kernel save folder.
        kernel_type: A specific class to force the model to load.
            Useful for custom kernels which don't appear in this module.

    """
    with config_path(directory).open("r") as f:
        config = json.load(f)
    params = np.load(params_path(directory))

    if kernel_type:
        raise NotImplementedError()
    else:
        kernel_type = config["type"]
        try:
            to_load = KERNEL_TYPES[kernel_type]
        except KeyError:
            raise KeyError(
                f"Found no kernel with type {kernel_type}. "
                "Specify a reference to the kernel's class using the `kernel_type` argument."
            )

        kernel = to_load()
        kernel.set_weights(params)
    return kernel
