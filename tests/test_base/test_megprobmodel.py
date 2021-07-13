"""Tests for the MEGNetProbModel class.
TODO: Add some pre-trained models for testing.
"""
from collections import deque
from pathlib import Path

import megnet
import numpy as np
import pandas as pd
import pytest
import unlockgnn
from pymatgen.util.testing import PymatgenTest
from unlockgnn import MEGNetProbModel

prob_model_parameters = ["gp_type, n_inducing", [("VGP", 10), ("GP", None)]]


@pytest.fixture
def phonons_model_1() -> MEGNetProbModel:
    """Get a model that has been trained to stage 1 on phonons data."""
    source = Path(__file__).parents[1] / "static" / "matbench_phonons" / "stage_1"
    return MEGNetProbModel.load(source)


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


def test_mutation(tmp_path, phonons_model_1):
    """Test that mutation preserves model weights."""
    new_save_dir = tmp_path / "phonons_mutated"
    kernel = phonons_model_1.kernel

    new_model = phonons_model_1.change_kernel_type(kernel, new_save_dir)

    assert new_model.training_stage == phonons_model_1.training_stage

    new_model.save()
    reloaded_new_model = MEGNetProbModel.load(new_save_dir)

    origin_weights = phonons_model_1.gnn.get_weights()
    new_weights = reloaded_new_model.gnn.get_weights()
    assert all(
        np.allclose(new_weight, origin_weight)
        for new_weight, origin_weight in zip(new_weights, origin_weights)
    )


@pytest.mark.slow
@pytest.mark.parametrize(*prob_model_parameters)
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
    NEW_GP_SAVE_DIR = SAVE_DIR / "new_gp"

    new_kernel_layer = unlockgnn.gp.kernel_layers.MaternOneHalfFn()
    # Need to extract the kernel property if our new model is a GP
    if gp_type == "GP":
        new_kernel = new_kernel_layer
        new_num_inducing_points = 10
    else:
        new_kernel = new_kernel_layer.kernel
        new_num_inducing_points = None

    new_gp_model = prob_model.change_gp_type(
        new_kernel, NEW_GP_SAVE_DIR, new_num_inducing_points
    )

    deque(new_gp_model.train_uq(epochs=1), maxlen=0)
