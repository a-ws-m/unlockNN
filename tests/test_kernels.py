"""Test the `kernel_layers` package."""
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from pathlib import Path
from typing import Type

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp
from unlocknn.kernel_layers import *

class ExampleKernel(KernelLayer):
    """An example kernel for testing."""

    def __init__(self, bias_variance: float=1.0, **kwargs):
        """Initialize the bias_variance parameter."""
        super().__init__(**kwargs)
        dtype = kwargs.get("dtype", tf.float64)
        self.bias_variance = self.add_weight(initializer=tf.constant_initializer(bias_variance), dtype=dtype, name="bias_variance")

    def kernel(self) -> tfp.math.psd_kernels.PositiveSemidefiniteKernel:
        """Get a linear kernel that's parameterised by a given bias variance."""
        return tfp.math.psd_kernels.Linear(bias_variance=self.bias_variance)
    
    @property
    def config(self) -> dict:
        """Return an empty dict: no kwargs are required."""
        return dict()

def test_custom_kernel_reloading(tmp_path: Path):
    """Test saving and loading with a custom kernel."""
    save_path = tmp_path / "kernel"
    kernel = ExampleKernel(bias_variance=2.0)
    kernel.save(save_path)
    reload_kernel = load_kernel(save_path, ExampleKernel)
    assert reload_kernel.get_weights()[0] == pytest.approx(2.0)


@pytest.mark.parametrize("kernel_type", [RBFKernelFn, MaternOneHalfFn])
def test_reload(tmp_path: Path, kernel_type: Type[AmpAndLengthScaleFn]):
    """Test saving and reloading a builtin kernel to/from disk."""
    # Example layer weights: [_amplitude_basis, _length_scale_basis]
    example_weights = [np.array(2.0), np.array(3.0)]
    save_dir = tmp_path / "kernel"
    orig_kernel = kernel_type()
    orig_kernel.set_weights(example_weights)
    orig_kernel.save(save_dir)

    orig_amplitude = orig_kernel.amplitude
    orig_length_scale = orig_kernel.length_scale

    loaded_kernel: AmpAndLengthScaleFn = load_kernel(save_dir)
    assert isinstance(loaded_kernel, kernel_type)
    loaded_weights = loaded_kernel.get_weights()
    assert loaded_weights == example_weights

    loaded_amp = loaded_kernel.amplitude.numpy()
    loaded_ls = loaded_kernel.length_scale.numpy()
    assert loaded_amp == pytest.approx(orig_amplitude)
    assert loaded_ls == pytest.approx(orig_length_scale)
