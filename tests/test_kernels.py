"""Test the `kernel_layers` package."""
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from pathlib import Path
from typing import Type

import numpy as np
import pytest
import tensorflow_probability as tfp
from unlocknn.kernel_layers import *

class TestKernel(KernelLayer):
    """An example kernel for testing."""

    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, bias_variance=1.0, **kwargs):
        """Initialize the bias_variance parameter."""
        self.bias_variance = bias_variance
        super().__init__(trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs)

    def kernel(self) -> tfp.math.psd_kernels.PositiveSemidefiniteKernel:
        """Get a linear kernel that's parameterised by a given bias variance."""
        return tfp.math.psd_kernels.Linear(self.bias_variance)
    
    @property
    def config(self) -> dict:
        """Return the bias_variance."""
        return {"bias_variance": self.bias_variance}

def test_custom_kernel_reloading(tmp_path: Path):
    """Test saving and loading with a custom kernel."""
    save_path = tmp_path / "kernel"
    kernel = TestKernel(bias_variance=2.0)
    kernel.save(save_path)
    reload_kernel = load_kernel(save_path, TestKernel)
    assert reload_kernel.kernel().bias_variance.numpy().item() == pytest.approx(2.0)


@pytest.mark.parametrize("kernel_type", [RBFKernelFn, MaternOneHalfFn])
def test_reload(tmp_path: Path, kernel_type: Type[AmpAndLengthScaleFn]):
    """Test saving and reloading a builtin kernel to/from disk."""
    # Example layer weights: [_amplitude, _length_scale]
    example_weights = [np.array(2.0), np.array(3.0)]
    save_dir = tmp_path / "kernel"
    orig_kernel = kernel_type()
    orig_kernel.set_weights(example_weights)
    orig_kernel.save(save_dir)

    loaded_kernel = load_kernel(save_dir)
    assert isinstance(loaded_kernel, kernel_type)
    loaded_weights = loaded_kernel.get_weights()
    assert loaded_weights == example_weights
