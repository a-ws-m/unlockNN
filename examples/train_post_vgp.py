"""Train a VGP on pre-existing MEGNet layer outputs."""
import typing

try:
    import mlflow

    track = True
except ImportError:
    # No mlflow tracking
    track = False

import numpy as np
import pandas as pd
import tensorflow as tf
from pyarrow import feather

from sse_gnn.gp.vgp_trainer import SingleLayerVGP
from sse_gnn.utilities import deserialize_array

from .config import DB_DIR, MODELS_DIR

checkpoint_path = str(MODELS_DIR / "vgp_ckpts.{epoch:02d}-{val_loss:.4f}.h5")

dtype = tf.float64
train_df = feather.read_feather(DB_DIR / "gp_train_df.fthr")
test_df = feather.read_feather(DB_DIR / "gp_test_df.fthr")

obs_layers = np.stack(train_df["layer_out"].apply(deserialize_array))
observation_index_points = tf.constant(obs_layers, dtype=dtype)

val_layers = np.stack(test_df["layer_out"].apply(deserialize_array))
index_points = tf.constant(val_layers, dtype=dtype)


def get_sses(df: pd.DataFrame) -> np.ndarray:
    """Get stacked SSE values."""
    cat_sses = df["cat_sse"].apply(deserialize_array)
    an_sses = df["an_sse"].apply(deserialize_array)
    return np.column_stack((cat_sses, an_sses))


train_sses = tf.constant(get_sses(train_df), dtype=dtype)
test_sses = tf.constant(get_sses(test_df), dtype=dtype)

vgp = SingleLayerVGP(observation_index_points, ntargets=2)


def model_train():
    """Run the training sequence."""
    vgp.train_model(
        train_sses,
        (index_points, test_sses),
        epochs=100,
        checkpoint_path=checkpoint_path,
    )


if track:
    with mlflow.start_run():
        mlflow.tensorflow.autolog(1)
        model_train()
else:
    model_train()

vgp.model.save_weights(str(MODELS_DIR / "vgp_v1.h5"))
