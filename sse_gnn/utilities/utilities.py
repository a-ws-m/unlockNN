"""Utility code snippets."""
import mlflow
from tensorflow.keras.callbacks import Callback


class MLFlowMetricsLogger(Callback):
    """Handler for logging metrics to MLFlow more cleanly."""

    def on_epoch_end(self, epoch, logs=None):
        """Log metrics to MLFlow."""
        if logs is None:
            return
        mlflow.log_metrics(logs, epoch)
