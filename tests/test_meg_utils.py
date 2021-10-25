"""Test for the MEGNet utilities module"""
import os


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from pathlib import Path

import numpy as np
import pytest
from megnet.models.megnet import MEGNetModel
from megnet.utils.preprocessing import Scaler
from unlocknn.megnet_utils import create_megnet_input

from .utils import datadir, load_df_head


class ExampleScaler(Scaler):
    """An example scaler for testing.
    
    Performs transforming by dividing by number of atoms,
    and inverse transforming by multiplying.
    
    """
    def transform(self, target: np.ndarray, n: int = 1) -> np.ndarray:
        return target / n
    
    def inverse_transform(self, transformed_target: np.ndarray, n: int = 1) -> np.ndarray:
        return transformed_target * n

def test_input_with_scaler(datadir: Path):
    """Test input generation."""
    binary_dir = datadir / "mp_binary_on_hull.pkl"
    binary_df = load_df_head(binary_dir)
    
    meg_model = MEGNetModel.from_file(str(datadir / "formation_energy.hdf5"))
    meg_model.target_scaler = ExampleScaler()

    input_gen, _ = create_megnet_input(
        meg_model,
        binary_df["structure"],
        binary_df["formation_energy_per_atom"],
        batch_size=100,  # We have just one batch
        shuffle=False
    )

    # Get first batch (the whole input)
    _, scaled_targets = input_gen.__getitem__(0)
    # Check targets are scaled
    scaled_targets == pytest.approx(binary_df["formation_energy_per_atom"] / binary_df["num_atoms"], rel=1e-6)
