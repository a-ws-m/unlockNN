"""Tests for the MEGNetProbModel class."""
import megnet
import numpy as np
import pytest
from matminer.datasets import load_dataset
from megnet.data.crystal import CrystalGraph
from sklearn.model_selection import train_test_split

import unlockgnn
from unlockgnn import MEGNetProbModel


@pytest.mark.slow
@pytest.mark.parametrize("gp_type, n_inducing", [("VGP", 10), ("GP", None)])
def test_train(tmp_path, mocker, gp_type, n_inducing):
    """Test training of the joint model using a benchmark dataset."""
    # Patch out training routines to speed up process
    mocker.patch("megnet.models.MEGNetModel.train")

    # gp_train_func = "unlockgnn.gp.{}.train_model".format(
    #     "gp_trainer.GPTrainer" if gp_type == "GP" else "vgp_trainer.SingleLayerVGP"
    # )
    # mocker.patch(gp_train_func)

    # Dataset information
    SAVE_DIR = tmp_path
    DATASET = "dielectric_constant"
    TARGET_VAR = "band_gap"

    data = load_dataset(DATASET, data_home=str(SAVE_DIR))

    train_df, test_df = train_test_split(data, random_state=2020)

    # Standard MEGNet arguments
    nfeat_bond = 10
    r_cutoff = 5
    gaussian_centers = np.linspace(0, r_cutoff + 1, nfeat_bond)
    gaussian_width = 0.5
    graph_converter = CrystalGraph(cutoff=r_cutoff)
    meg_args = {
        "graph_converter": graph_converter,
        "centers": gaussian_centers,
        "width": gaussian_width,
        "metrics": ["mae"],
    }

    # Model creation
    prob_model = MEGNetProbModel(
        train_df["structure"],
        train_df[TARGET_VAR],
        test_df["structure"],
        test_df[TARGET_VAR],
        gp_type,
        SAVE_DIR,
        num_inducing_points=n_inducing,
        **meg_args,
    )

    prob_model.train_gnn(epochs=1)
    megnet.models.MEGNetModel.train.assert_called_once()

    prob_model.train_uq(epochs=1)
    # eval(gp_train_func).assert_called_once()

    prob_model.save(train_df.index, test_df.index)

    reload = MEGNetProbModel.load(tmp_path)
