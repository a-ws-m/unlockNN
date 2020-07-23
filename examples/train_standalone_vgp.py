"""Train model to determine SSEs using MEGNet."""
try:
    import mlflow

    track = True
except ImportError:
    # No tracking
    track = False

import numpy as np
import pyarrow.feather as feather
import pymatgen
from megnet.data.crystal import CrystalGraph
from sklearn.model_selection import train_test_split

from sse_gnn.standalone import ProbabilisticMEGNetModel

from .config import DB_DIR, SSE_DB_LOC

nfeat_bond = 10
r_cutoff = 5
gaussian_centers = np.linspace(0, r_cutoff + 1, nfeat_bond)
gaussian_width = 0.5
graph_converter = CrystalGraph(cutoff=r_cutoff)
prob_model = ProbabilisticMEGNetModel(
    graph_converter=graph_converter,
    centers=gaussian_centers,
    width=gaussian_width,
    ntarget=2,
    metrics=["mae"],
    batch_size=128,
)

# * Get structure data
data = feather.read_feather(SSE_DB_LOC)

structures = [
    pymatgen.Structure.from_str(struct, "json") for struct in data["structure"]
]
sse_tuples = list(zip(data["cat_sse"].values, data["an_sse"].values))
sse_vectors = [np.array(sse_pair) for sse_pair in sse_tuples]

train_structs, val_structs, train_sses, val_sses = train_test_split(
    structures, sse_vectors, random_state=2020
)

# * Save the split we train with for use in the GP
train_struc_strs = [struct.to("json") for struct in train_structs]
test_struc_strs = [struct.to("json") for struct in val_structs]

train_df = data.loc[data["structure"].isin(train_struc_strs)]
test_df = data.loc[data["structure"].isin(test_struc_strs)]

feather.write_feather(train_df, DB_DIR / "train_df.fthr")
feather.write_feather(test_df, DB_DIR / "test_df.fthr")

# * Model training


def model_train():
    """Train the model."""
    # Here, `structures` is a list of pymatgen Structure objects.
    # `targets` is a corresponding list of properties.
    prob_model.train_from_structs(
        train_structs, train_sses, val_structs, val_sses, epochs=10
    )


if track:
    with mlflow.start_run(run_name="laptop_test_sequence"):
        mlflow.tensorflow.autolog(every_n_iter=1)
        model_train()
else:
    model_train()

prob_model.model.save_model("megnet_model_v1")
