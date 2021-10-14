"""Test downloading suite."""
import os
from pathlib import Path

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# See https://github.com/tensorflow/tensorflow/issues/152#issuecomment-273663277
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from unlocknn.download import load_data, load_pretrained


def test_load_model(tmp_path: Path):
    """Test downloading formation energies model."""
    orig_model = load_pretrained("binary_e_form", save_dir=tmp_path)
    reload_model = load_pretrained("binary_e_form", save_dir=tmp_path)


def test_load_data(tmp_path: Path):
    """Test downloading formation energies data."""
    orig_data = load_data("binary_e_form", save_dir=tmp_path)
    reload_data = load_data("binary_e_form", save_dir=tmp_path)

    assert orig_data.loc[0, "formation_energy_per_atom"] == -0.7374389025000001
    assert reload_data.loc[0, "formation_energy_per_atom"] == -0.7374389025000001
