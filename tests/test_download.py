"""Test downloading suite."""
import os
from pathlib import Path

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from unlockgnn.download import load_pretrained


def test_load(tmp_path: Path):
    """Test downloading phonons dataset."""
    orig_model = load_pretrained("binary_e_form", branch="urop-2021", save_dir=tmp_path)
    reload_model = load_pretrained(
        "binary_e_form", branch="urop-2021", save_dir=tmp_path
    )
