"""Train a VGP on pre-existing MEGNet layer outputs."""
try:
    import mlflow

    track = True
except ImportError:
    # No mlflow tracking
    track = False

import numpy as np
import pandas as pd
import tensorflow as tf

from sse_gnn.gp.vgp_trainer import SingleLayerVGP

from .config import DB_DIR, MODELS_DIR

checkpoint_path = str(MODELS_DIR / "vgp_ckpts.{epoch:02d}-{val_loss:.4f}.h5")

dtype = tf.float64
train_df = pd.read_pickle(DB_DIR / "gp_train_df.pickle")
test_df = pd.read_pickle(DB_DIR / "gp_test_df.pickle")

observation_index_points = tf.constant(
    np.stack(train_df["layer_out"].values), dtype=dtype
)
index_points = tf.constant(np.stack(test_df["layer_out"].values), dtype=dtype)

train_sses = tf.constant(np.array(list(train_df["sses"].values)), dtype=dtype)
test_sses = tf.constant(np.array(list(test_df["sses"].values)), dtype=dtype)

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
