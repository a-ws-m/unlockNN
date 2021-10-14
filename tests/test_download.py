"""Test downloading suite."""
import os
from pathlib import Path

import pytest
import requests
from pytest_mock import MockerFixture

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# See https://github.com/tensorflow/tensorflow/issues/152#issuecomment-273663277
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from unlocknn.download import load_data, load_pretrained


def test_load_model(tmp_path: Path, mocker: MockerFixture):
    """Test downloading formation energies model."""
    get_spy = mocker.spy(requests, "get")
    load_pretrained("binary_e_form", save_dir=tmp_path)
    get_spy.assert_called_once()
    load_pretrained("binary_e_form", save_dir=tmp_path)
    # Second call should load from disk
    get_spy.assert_called_once()


def test_load_data(tmp_path: Path):
    """Test downloading formation energies data."""
    orig_data = load_data("binary_e_form", save_dir=tmp_path)
    reload_data = load_data("binary_e_form", save_dir=tmp_path)

    assert orig_data.loc[0, "formation_energy_per_atom"] == pytest.approx(-0.7374389025, abs=1e-10)
    assert reload_data.loc[0, "formation_energy_per_atom"] == pytest.approx(-0.7374389025, abs=1e-10)
