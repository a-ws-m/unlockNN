"""Analyse metrics and generate plots for a trained GP."""
import typing
from operator import attrgetter

import numpy as np
import pandas as pd
import tensorflow as tf
from pyarrow import feather

from sse_gnn.datalib.metrics import MetricAnalyser
from sse_gnn.gp.vgp_trainer import SingleLayerVGP
from sse_gnn.utilities import deserialize_array

from .config import DB_DIR, MODELS_DIR

prev_model = str(MODELS_DIR / "post_vgp" / "vgp_v2.h5")

SHARPNESS_FILE = "vgp_sharpness.png"
CALIBRATION_FILE = "vgp_calibration.png"

dtype = tf.float64
train_df = feather.read_feather(DB_DIR / "gp_train_df.fthr")
test_df = feather.read_feather(DB_DIR / "gp_test_df.fthr")

obs_layers = np.stack(train_df["layer_out"].apply(deserialize_array))
observation_index_points = tf.constant(obs_layers, dtype=dtype)

val_layers = np.stack(test_df["layer_out"].apply(deserialize_array))
index_points = tf.constant(val_layers, dtype=dtype)


def get_sses(df: pd.DataFrame) -> np.ndarray:
    """Get stacked SSE values."""
    return np.column_stack((df["cat_sse"], df["an_sse"]))


train_sses = tf.constant(get_sses(train_df), dtype=dtype)
test_sses = tf.constant(get_sses(test_df), dtype=dtype)

vgp = SingleLayerVGP(observation_index_points, ntargets=2, prev_model=prev_model)

ma = MetricAnalyser(
    index_points, test_sses, vgp(index_points), calc_stddev_on_init=False
)

metrics_ls = ["mae", "nll"]
metrics = {metric: attrgetter(metric)(ma) for metric in metrics_ls}
print(metrics)
print(f"{ma.mean.shape=}")
print(f"{tf.losses.mae(tf.transpose(test_sses), ma.mean.T)=}")

# ma.sharpness_plot(SHARPNESS_FILE)
# ma.calibration_plot(CALIBRATION_FILE)
