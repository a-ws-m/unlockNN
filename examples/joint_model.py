from pathlib import Path

import numpy as np
from matminer.datasets import get_all_dataset_info, load_dataset
from megnet.data.crystal import CrystalGraph
from sklearn.model_selection import train_test_split
from sse_gnn import MEGNetProbModel

# from sse_gnn.datalib.metrics import MetricAnalyser


SAVE_DIR = Path.home() / "matbench_perovskites"
DATASET = "matbench_perovskites"
TARGET_VAR = "e_form"

print(get_all_dataset_info(DATASET))

data = load_dataset(DATASET)

train_df, test_df = train_test_split(data, random_state=2020)

print(f"Train DF:\n{train_df.describe()}")
print(f"Test DF:\n{test_df.describe()}")

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
    num_inducing_points=200,
    **meg_args,
)

print("Training MEGNetModel")
prob_model.train_meg_model(epochs=100)

print("Training UQ")
prob_model.train_uq(epochs=100)

print("Saving model")
prob_model.save(train_df.index, test_df.index)
