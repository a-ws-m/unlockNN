"""Train GP from MEGNet output data."""
try:
    import mlflow

    track = True
except ImportError:
    # No mlflow tracking
    track = False

import numpy as np
import pyarrow.feather as feather
import tensorflow as tf

from unlockgnn.datalib.metrics import MetricAnalyser
from unlockgnn.gp.gp_trainer import GPTrainer, convert_index_points
from unlockgnn.utilities import deserialize_array

from .config import (
    CKPT_PATH,
    DB_DIR,
    MODELS_DIR,
    N_EPOCHS,
    NEW_MODEL,
    PATIENCE,
    RUN_NAME,
)

SHARPNESS_FILE = "gp_sharpness.png"
CALIBRATION_FILE = "gp_calibration.png"

# * Get the data and perform some regression

train_df = feather.read_feather(DB_DIR / "gp_train_df.fthr")
test_df = feather.read_feather(DB_DIR / "gp_test_df.fthr")

observation_index_points = np.stack(train_df["layer_out"].apply(deserialize_array))
index_points = np.stack(test_df["layer_out"].apply(deserialize_array))

observation_index_points = convert_index_points(observation_index_points)
index_points = convert_index_points(index_points)

cat_observations = tf.constant(train_df["cat_sse"], dtype=tf.float64)
# an_observations = tf.constant(train_df["an_sse"], dtype=tf.float64)

cat_test_vals = tf.constant(test_df["cat_sse"], dtype=tf.float64)
# an_test_vals = tf.constant(test_df["an_sse"], dtype=tf.float64)

metric_labels = [
    "nll",
    "mae",
    "sharpness",
    "variation",
    "calibration_err",
]

model_params = {
    "epochs": N_EPOCHS,
    "patience": PATIENCE,
    "metrics": metric_labels,
    "save_dir": str(MODELS_DIR / "post_gp" / NEW_MODEL),
}

# * Build cation SSE GP model
cat_gp_trainer = GPTrainer(
    observation_index_points,
    cat_observations,
    str(MODELS_DIR / "post_gp" / CKPT_PATH) if CKPT_PATH else None,
)

# * Get current model metrics
gp_metrics = MetricAnalyser(
    index_points, cat_test_vals, cat_gp_trainer.get_model(index_points)
)
for metric in metric_labels:
    print(f"{metric}: {getattr(gp_metrics, metric)}")

# * Plot sharpness and calibration error
gp_metrics.sharpness_plot(SHARPNESS_FILE)
gp_metrics.calibration_plot(CALIBRATION_FILE)

# * Train model
if track:
    with mlflow.start_run(run_name=RUN_NAME) as run:
        for step, metric_dict in enumerate(
            cat_gp_trainer.train_model(index_points, cat_test_vals, **model_params)  # type: ignore
        ):
            mlflow.log_metrics(metric_dict, step + 1)

else:
    metrics = [
        metric_dict
        for metric_dict in cat_gp_trainer.train_model(
            index_points, cat_test_vals, **model_params  # type: ignore
        )
    ]

# * Print model parameters
print(f"{cat_gp_trainer.trainable_vars=}")
