"""Train a VGP on pre-existing MEGNet layer outputs."""
import numpy as np
import pandas as pd
import tensorflow as tf
from pyarrow import feather

from unlockgnn.gp.vgp_trainer import SingleLayerVGP
from unlockgnn.utilities import deserialize_array, MLFlowMetricsLogger

from .config import (
    CKPT_PATH,
    DB_DIR,
    MODELS_DIR,
    N_EPOCHS,
    NEW_MODEL,
    NUM_INDUCING,
    PATIENCE,
    PREV_MODEL,
    RUN_NAME,
)

try:
    import mlflow

    track = True
except ImportError:
    # No mlflow tracking
    track = False


assert NUM_INDUCING is not None

checkpoint_path = str(MODELS_DIR / "post_vgp" / CKPT_PATH) if CKPT_PATH else None

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

vgp = SingleLayerVGP(
    observation_index_points, NUM_INDUCING, ntargets=2, prev_model=PREV_MODEL
)


def model_train(callbacks=[]):
    """Run the training sequence."""
    vgp.train_model(
        train_sses,
        (index_points, test_sses),
        epochs=N_EPOCHS,
        patience=PATIENCE,
        checkpoint_path=checkpoint_path,
        callbacks=callbacks,
    )


if track:
    callbacks = [MLFlowMetricsLogger()]
    with mlflow.start_run(run_name=RUN_NAME):
        model_train(callbacks)
else:
    model_train()

vgp.model.save_weights(str(MODELS_DIR / NEW_MODEL))
