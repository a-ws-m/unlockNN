"""Tests for the MEGNetProbModel class."""
from collections import deque

import megnet
import pandas as pd
import pymatgen
import pytest
from pathlib import Path
from pymatgen.util.testing import PymatgenTest

import unlockgnn
from unlockgnn import MEGNetProbModel


@pytest.fixture
def structure_database() -> pd.DataFrame:
    """Get a minimal test database for training."""
    # Based on the first results from the materials project
    # The true values don't really matter for testing
    structure_band_gaps = {"TiO2": 2.694, "SiO2": 5.851, "VO2": 0, "Li2O2": 2.023}
    data = {
        "structure": [
            PymatgenTest.get_structure(name) for name in structure_band_gaps.keys()
        ],
        "band_gap": list(structure_band_gaps.values()),
    }

    return pd.DataFrame(data)


@pytest.mark.slow
@pytest.mark.parametrize("gp_type, n_inducing", [("VGP", 10), ("GP", None)])
def test_train(tmp_path, mocker, structure_database, gp_type, n_inducing):
    """Test training of the joint model using a benchmark dataset.

    This is an integration test to catch any glaring errors.

    """
    # Patch out training routines to speed up process
    mocker.patch("megnet.models.MEGNetModel.train")

    # gp_train_func = "unlockgnn.gp.{}.train_model".format(
    #     "gp_trainer.GPTrainer" if gp_type == "GP" else "vgp_trainer.SingleLayerVGP"
    # )
    # mocker.patch(gp_train_func)

    # Dataset information
    SAVE_DIR = Path(tmp_path)
    TARGET_VAR = "band_gap"

    data = structure_database

    train_df = data.iloc[:2, :]
    test_df = data.iloc[2:, :]

    # * Standard MEGNet arguments, now not needed
    # nfeat_bond = 10
    # r_cutoff = 5
    # gaussian_centers = np.linspace(0, r_cutoff + 1, nfeat_bond)
    # gaussian_width = 0.5
    # graph_converter = CrystalGraph(cutoff=r_cutoff)
    # meg_args = {
    #     "graph_converter": graph_converter,
    #     "centers": gaussian_centers,
    #     "width": gaussian_width,
    #     "metrics": ["mae"],
    # }

    # Model creation
    prob_model = MEGNetProbModel(
        train_df["structure"],
        train_df[TARGET_VAR],
        test_df["structure"],
        test_df[TARGET_VAR],
        gp_type,
        SAVE_DIR,
        num_inducing_points=n_inducing,
    )

    prob_model.train_gnn(epochs=1)
    megnet.models.MEGNetModel.train.assert_called_once()

    # Exhaust iterator
    deque(prob_model.train_uq(epochs=1), maxlen=0)
    # eval(gp_train_func).assert_called_once()

    # * Test IO routine
    prob_model.save(train_df.index, test_df.index)
    prob_model = MEGNetProbModel.load(SAVE_DIR)

    # * Test mutation operations
    NEW_SAVE_DIR = SAVE_DIR / "new_kernel"

    new_kernel_layer = unlockgnn.gp.kernel_layers.MaternOneHalfFn()
    # Need to extract the kernel property if we're using a GP
    new_kernel = new_kernel_layer if gp_type == "VGP" else new_kernel_layer.kernel

    new_kernel_model = prob_model.change_kernel_type(new_kernel, NEW_SAVE_DIR)

    deque(new_kernel_model.train_uq(epochs=1), maxlen=0)

    new_kernel_model.save()
    MEGNetProbModel.load(NEW_SAVE_DIR)
