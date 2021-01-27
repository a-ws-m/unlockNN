"""Kernel layer implementations for use with VariationalGaussianProcesses."""
from abc import ABC, abstractmethod

import tensorflow as tf
import tensorflow_probability as tfp


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


class AmpAndLengthScaleFn(KernelLayer):
    """An ABC for kernels with amplitude and length scale parameters.

    Attributes:
        _amplitude (tf.Tensor): The amplitude of the kernel.
        _length_scale (tf.Tensor): The length scale of the kernel.

    """

    def __init__(self, **kwargs):
        """Initialize layer and parameters."""
        super().__init__(**kwargs)
        dtype = kwargs.get("dtype", None)

        self._amplitude = self.add_variable(
            initializer=tf.constant_initializer(0), dtype=dtype, name="amplitude"
        )

        self._length_scale = self.add_variable(
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