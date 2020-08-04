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
from sse_gnn.utilities import MLFlowMetricsLogger

from .config import (
    CKPT_PATH,
    N_EPOCHS,
    NEW_MODEL,
    NUM_INDUCING,
    PATIENCE,
    RUN_NAME,
    SSE_DB_LOC,
)

assert NUM_INDUCING is not None

nfeat_bond = 10
r_cutoff = 5
gaussian_centers = np.linspace(0, r_cutoff + 1, nfeat_bond)
gaussian_width = 0.5
graph_converter = CrystalGraph(cutoff=r_cutoff)
prob_model = ProbabilisticMEGNetModel(
    NUM_INDUCING,
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

# * Model training


def model_train(callbacks=[]):
    """Train the model."""
    prob_model.train_from_structs(
        train_structs,
        train_sses,
        val_structs,
        val_sses,
        epochs=N_EPOCHS,
        checkpoint_path=CKPT_PATH,
        patience=PATIENCE,
        callbacks=callbacks,
    )


if track:
    callbacks = [MLFlowMetricsLogger()]
    with mlflow.start_run(run_name=RUN_NAME):
        model_train(callbacks)
else:
    model_train()

prob_model.model.save_model(NEW_MODEL)
