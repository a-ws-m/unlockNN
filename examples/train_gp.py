"""Train GP from MEGNet output data."""
from operator import itemgetter

try:
    import mlflow

    track = True
except ImportError:
    # No mlflow tracking
    track = False

import numpy as np
import pyarrow.feather as feather
import tensorflow as tf

from sse_gnn.gp.gp_trainer import GPMetrics, GPTrainer, convert_index_points
from sse_gnn.utilities import deserialize_array

from .config import DB_DIR, MODELS_DIR

# * Get the data and perform some regression

train_df = feather.read_feather(DB_DIR / "gp_train_df.fthr")
test_df = feather.read_feather(DB_DIR / "gp_test_df.fthr")

observation_index_points = np.stack(train_df["layer_out"].apply(deserialize_array))
index_points = np.stack(test_df["layer_out"].apply(deserialize_array))

observation_index_points = convert_index_points(observation_index_points)
index_points = convert_index_points(index_points)

cation_sses = train_df["cat_sse"].apply(deserialize_array)
# anion_sses = train_df["an_sse"].apply(deserialize_array)

cat_observations = tf.constant(cation_sses, dtype=tf.float64)
# an_observations = tf.constant(anion_sses, dtype=tf.float64)

cat_test_vals = tf.constant(test_df["cat_sse"].apply(deserialize_array))
# an_test_vals = tf.constant(test_df["an_sse"].apply(deserialize_array))

metric_labels = [
    "nll",
    "mae",
    "sharpness",
    "variation",
    # "calibration_err",
]

model_params = {
    "epochs": 200,
    "patience": 200,
    "metrics": metric_labels,
    "save_dir": str(MODELS_DIR / "saved_gp"),
}

# * Build cation SSE GP model
cat_gp_trainer = GPTrainer(
    observation_index_points, cat_observations, str(MODELS_DIR / "tf_ckpts")
)

# * Get current model metrics
gp_metrics = GPMetrics(index_points, cat_test_vals, cat_gp_trainer)
for metric in metric_labels:
    print(f"{metric}: {getattr(gp_metrics, metric)}")

# * Plot sharpness and calibration error
gp_metrics.sharpness_plot()
gp_metrics.calibration_plot()

# * Train model
if track:
    with mlflow.start_run(run_name="laptop_test") as run:
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
print(f"{cat_gp_trainer.amplitude=}")
print(f"{cat_gp_trainer.length_scale=}")
