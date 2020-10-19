"""Example implementation of a MEGNetProbModel."""
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from matminer.datasets import get_all_dataset_info, load_dataset
from megnet.data.crystal import CrystalGraph
from sklearn.model_selection import train_test_split
from unlockgnn import MEGNetProbModel

SAVE_DIR = Path.home() / "matbench_perovskites"
DATASET = "matbench_perovskites"
TARGET_VAR = "e_form"

GNN_TRAINING_EPOCHS: int = 100
UQ_TRAINING_EPOCHS: int = 100
UQ_TYPE: Literal["GP", "VGP"] = "VGP"
NUM_INDUCING_POINTS: Optional[int] = 200  # Must only be set for "VGP"

print(get_all_dataset_info(DATASET))
data = load_dataset(DATASET)
train_df, test_df = train_test_split(data, random_state=2020)

print(f"Train DF:\n{train_df.describe()}")
print(f"Test DF:\n{test_df.describe()}")

# MEGNet parameters for crystal-to-graph conversion
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

# Model construction
prob_model = MEGNetProbModel(
    train_df["structure"],
    train_df[TARGET_VAR],
    test_df["structure"],
    test_df[TARGET_VAR],
    UQ_TYPE,
    SAVE_DIR,
    num_inducing_points=NUM_INDUCING_POINTS,
    **meg_args,
)

print("Training MEGNetModel")
prob_model.train_meg_model(epochs=GNN_TRAINING_EPOCHS)

print("Training UQ")
prob_model.train_uq(epochs=UQ_TRAINING_EPOCHS)

print("Saving model")
prob_model.save(train_df.index, test_df.index)
