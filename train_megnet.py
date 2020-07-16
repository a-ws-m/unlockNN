"""Train model to determine SSEs using MEGNet."""
import mlflow
import mlflow.tensorflow
import numpy as np
import pandas as pd
import pyarrow.feather as feather
import pymatgen
from megnet.data.crystal import CrystalGraph
from megnet.models import MEGNetModel
from sklearn.model_selection import train_test_split

from .config import *

mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Users/awm1017@ic.ac.uk/MEGNet SSE Model")

nfeat_bond = 10
r_cutoff = 5
gaussian_centers = np.linspace(0, r_cutoff + 1, nfeat_bond)
gaussian_width = 0.5
graph_converter = CrystalGraph(cutoff=r_cutoff)
model = MEGNetModel(
    graph_converter=graph_converter,
    centers=gaussian_centers,
    width=gaussian_width,
    ntarget=2,
    metrics=["mse", "mape"],
)

# * Get structure data
data = feather.read_feather(SSE_DB_LOC)

structures = [
    pymatgen.Structure.from_str(struct, "json") for struct in data["structure"]
]
sse_tuples = list(zip(data["cat_sse"].values, data["an_sse"].values))
train_structs, test_structs, train_sses, test_sses = train_test_split(
    structures, sse_tuples, random_state=2020
)

# * Save the split we train with for use in the GP
# train_struc_strs = [struct.to("json") for struct in train_structs]
# test_struc_strs = [struct.to("json") for struct in test_structs]

# train_df = data.loc[data["structure"].isin(train_struc_strs)]
# test_df = data.loc[data["structure"].isin(test_struc_strs)]

# feather.write_feather(train_df, DB_DIR / "train_df.fthr")
# feather.write_feather(test_df, DB_DIR / "test_df.fthr")

# * Model training
# Here, `structures` is a list of pymatgen Structure objects.
# `targets` is a corresponding list of properties.
with mlflow.start_run(run_name="laptop_test"):
    mlflow.tensorflow.autolog(every_n_iter=1)
    model.train(train_structs, train_sses, test_structs, test_sses, epochs=10)

# model.save_model("megnet_model_v1")
