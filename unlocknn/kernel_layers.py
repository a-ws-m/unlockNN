"""Kernel layer implementations for use with VariationalGaussianProcesses."""
from abc import ABC, abstractmethod
import json
from os import PathLike, mkdir
from pathlib import Path
from typing import Optional, Type

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

    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        """Initialize kernel parameters.

        Subclasses should use :meth:`add_weight` to initialize kernel layer
        weights during initialization.

        """
        super().__init__(
            trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs
        )

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
        """Get configuration information for the class.

        This configuration information is used by :func:`load_kernel` to
        determine what type of kernel to load. For built-in kernels, this
        configuration dictionary specifies ``{"type": type}``, where ``type`` is
        one of the keys in :const:`KERNEL_TYPES`.

        For custom kernels, this dictionary should return any keyword arguments
        to be passed during kernel initialization.

        """
        ...

    def save(self, directory: PathLike):
        """Save the kernel to disk.

        Subclasses should update this method to save configuration information
        as well (c.f. :meth:`config`). See also :func:`load_kernel`.

        """
        directory = Path(directory)
        if not directory.exists():
            mkdir(directory)
        weights = self.get_weights()
        np.save(params_path(directory), weights, allow_pickle=True)

        with config_path(directory).open("w") as f:
            json.dump(self.config, f)


class AmpAndLengthScaleFn(KernelLayer, ABC):
    """An ABC for kernels with amplitude and length scale parameters.

    Attributes:
        _amplitude_basis (tf.Tensor): The basis for the kernel amplitude,
            which is passed through a softplus to calculate the actual amplitude.
        _length_scale_basis (tf.Tensor): The basis for the length scale of the kernel.
            which is passed through a softplus to calculate the actual amplitude.

    """

    def __init__(self, **kwargs):
        """Initialize the layer, its amplitude and its length scale."""
        super().__init__(**kwargs)
        dtype = kwargs.get("dtype", tf.float64)

        self._amplitude_basis = self.add_weight(
            initializer=tf.constant_initializer(0), dtype=dtype, name="amplitude"
        )

        self._length_scale_basis = self.add_weight(
            initializer=tf.constant_initializer(0), dtype=dtype, name="length_scale"
        )
    
    @property
    def amplitude(self) -> tf.Tensor:
        """Get the current kernel amplitude."""
        return tf.nn.softplus(0.1 * self._amplitude_basis)
    
    @property
    def length_scale(self) -> tf.Tensor:
        """Get the current kernel length scale."""
        return tf.nn.softplus(5.0 * self._length_scale_basis)


class RBFKernelFn(AmpAndLengthScaleFn):
    """A radial basis function implementation that works with keras.

    Attributes:
        _amplitude_basis (tf.Tensor): The basis for the kernel amplitude,
            which is passed through a softplus to calculate the actual amplitude.
        _length_scale_basis (tf.Tensor): The basis for the length scale of the kernel.
            which is passed through a softplus to calculate the actual amplitude.

    """

    @property
    def kernel(self) -> tfp.math.psd_kernels.PositiveSemidefiniteKernel:
        """Get a callable kernel."""
        return tfp.math.psd_kernels.ExponentiatedQuadratic(
            amplitude=self.amplitude,
            length_scale=self.length_scale,
        )

    @property
    def config(self) -> dict:
        return {"type": "rbf"}


class MaternOneHalfFn(AmpAndLengthScaleFn):
    """A Matern kernel with parameter 1/2 implementation that works with keras.

    Attributes:
        _amplitude_basis (tf.Tensor): The basis for the kernel amplitude,
            which is passed through a softplus to calculate the actual amplitude.
        _length_scale_basis (tf.Tensor): The basis for the length scale of the kernel.
            which is passed through a softplus to calculate the actual amplitude.

    """

    @property
    def kernel(self) -> tfp.math.psd_kernels.PositiveSemidefiniteKernel:
        """Get a callable kernel."""
        return tfp.math.psd_kernels.MaternOneHalf(
            amplitude=self.amplitude,
            length_scale=self.length_scale,
        )

    @property
    def config(self) -> dict:
        return {"type": "matern"}


KERNEL_TYPES = {"rbf": RBFKernelFn, "matern": MaternOneHalfFn}


def load_kernel(
    directory: PathLike, kernel_type: Optional[Type[KernelLayer]] = None
) -> KernelLayer:
    """Load a kernel from the disk.

    Args:
        directory: Path to the kernel save folder.
        kernel_type: A specific class to force the model to load.
            Useful for custom kernels which don't appear in this module.

    Examples:
        Create an :class:`RBFKernelFn`, save it, then reload from disk:

        >>> from tempfile import TemporaryDirectory
        >>> rbf = RBFKernelFn()
        >>> with TemporaryDirectory() as tmpdirname:
        ...     rbf.save(tmpdirname)
        ...     new_rbf = load_kernel(tmpdirname)

    """
    directory = Path(directory)
    with config_path(directory).open("r") as f:
        config = json.load(f)
    params = np.load(params_path(directory))

    if kernel_type:
        kernel = kernel_type(**config)
        kernel.set_weights(params)
    else:
        kernel_name = config["type"]
        try:
            to_load = KERNEL_TYPES[kernel_name]
        except KeyError:
            raise KeyError(
                f"Found no kernel with type {kernel_name}. "
                "Specify a reference to the kernel's class using the `kernel_type` argument."
            )

        kernel = to_load()
        kernel.set_weights(params)
    return kernel


if __name__ == "__main__":  # pragma: no cover
    import doctest

    doctest.testmod()
