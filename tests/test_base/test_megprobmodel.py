"""Tests for the MEGNetProbModel class."""
import numpy as np
import pytest
from matminer.datasets import load_dataset
from megnet.data.crystal import CrystalGraph
from sklearn.model_selection import train_test_split

from unlockgnn import MEGNetProbModel


@pytest.mark.slow
def test_train(tmp_path):
    """Test training of the joint model using a benchmark dataset."""
    SAVE_DIR = tmp_path
    DATASET = "matbench_perovskites"
    TARGET_VAR = "e_form"

    data = load_dataset(DATASET, data_home=str(SAVE_DIR))

    train_df, test_df = train_test_split(data, random_state=2020)

    nfeat_bond = 100
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

    prob_model = MEGNetProbModel(
        train_df["structure"],
        train_df[TARGET_VAR],
        test_df["structure"],
        test_df[TARGET_VAR],
        "VGP",
        SAVE_DIR,
        num_inducing_points=10,
        **meg_args,
    )

    print("Training MEGNetModel")
    prob_model.train_meg_model(epochs=1)

    print("Training UQ")
    prob_model.train_uq(epochs=1)

    print("Saving model")
    prob_model.save(train_df.index, test_df.index)
