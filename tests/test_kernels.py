"""Test the `kernel_layers` package."""
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from pathlib import Path
from typing import Type

import numpy as np
import pytest
from unlockgnn.kernel_layers import *


@pytest.mark.parametrize("kernel_type", [RBFKernelFn, MaternOneHalfFn])
def test_reload(tmp_path: Path, kernel_type: Type[AmpAndLengthScaleFn]):
    """Test saving and reloading a kernel to/from disk."""
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
