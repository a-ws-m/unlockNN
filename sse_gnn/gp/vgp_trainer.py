"""VariationalGaussianProcess single layer model."""
from typing import Optional, Tuple

import mlflow
import mlflow.tensorflow
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.keras.utils import losses_utils

from .config import DB_DIR, MODELS_DIR


class RBFKernelFn(tf.keras.layers.Layer):
    """A radial basis function implementation that works with keras.

    Attributes:
        _amplitude (tf.Tensor): The amplitude of the kernel.
        _length_scale (tf.Tensor): The length scale of the kernel.

    """

    def __init__(self, **kwargs):
        """Initialize layer and parameters."""
        super().__init__(**kwargs)
        dtype = kwargs.get("dtype", None)

        self._amplitude = self.add_variable(
            initializer=tf.constant_initializer(0), dtype=dtype, name="amplitude"
        )

        self._length_scale = self.add_variable(
            initializer=tf.constant_initializer(0), dtype=dtype, name="length_scale"
        )

    def call(self, x):
        """Do nothing -- a placeholder for keras."""
        # Never called -- this is just a layer so it can hold variables
        # in a way Keras understands.
        return x

    @property
    def kernel(self) -> tfp.math.psd_kernels.PositiveSemidefiniteKernel:
        """Get a callable kernel."""
        return tfp.math.psd_kernels.ExponentiatedQuadratic(
            amplitude=tf.nn.softplus(0.1 * self._amplitude),
            length_scale=tf.nn.softplus(5.0 * self._length_scale),
        )


class VariationalLoss(tf.keras.losses.Loss):
    """Implementation of variational loss using keras API."""

    def __init__(self, kl_weight, reduction=losses_utils.ReductionV2.AUTO, name=None):
        """Initialize loss function and KL divergence loss scaling factor."""
        self.kl_weight = kl_weight
        super().__init__(reduction=reduction, name=name)

    def call(self, y_true, predicted_distribution):
        """Calculate the variational loss."""
        return predicted_distribution.variational_loss(y_true, kl_weight=self.kl_weight)


class SingleLayerVGP:
    """A network with an input and a VGP.

    Args:
        observation_indices (:obj:`tf.Tensor`): The (training) observation index points (x data).
        ntargets (int): The number of parameters to be modelled.
        num_inducing_points (int): The number of inducing points for the :obj:`VariationalGaussianProcess`.
        batch_size (int): The training batch size.

    Attributes:
        observation_indices (:obj:`tf.Tensor`): The (training) observation index points (x data).
        batch_size (int): The training batch size.
        model (:obj:`Model`): The Keras model containing the obj:`VariationalGaussianProcess`.

    """

    def __init__(
        self,
        observation_indices: tf.Tensor,
        ntargets: int = 1,
        num_inducing_points: int = 96,
        batch_size: int = 32,
    ):
        """Initialize and compile model."""
        self.observation_indices = observation_indices
        self.batch_size = batch_size

        # * Set up model
        # TODO: Generalise input shape
        inputs = tf.keras.layers.Input(shape=(96,))
        output = tfp.layers.VariationalGaussianProcess(
            num_inducing_points,
            RBFKernelFn(dtype=tf.float64),
            event_shape=(ntargets,),
            jitter=1e-06,
            convert_to_tensor_fn=tfp.distributions.Distribution.mean,
        )(inputs)
        model = tf.keras.Model(inputs, output)

        # * Compile model
        # Determine KL divergence scaling factor
        kl_weight = np.array(batch_size, np.float64) / observation_indices.shape[0]
        loss = VariationalLoss(kl_weight, name="variational_loss")

        model.compile(optimizer=tf.optimizers.Adam(), loss=loss, metrics=["mae"])

        self.model = model

    def __call__(self, *args, **kwargs):
        """Call the embedded Keras model."""
        return self.model(*args, **kwargs)

    def train_model(
        self,
        observations: tf.Tensor,
        validation_data: Optional[Tuple] = None,
        epochs: int = 1000,
    ):
        """Train the model.

        Args:
            observations (:obj:`tf.Tensor`): The observed true _y_ values.
            validation_data (tuple of :obj:`tf.Tensor`, optional): The validation data
                as a tuple of (validation_x, validation_y).
            epochs (int): The number of training epochs.

        """
        checkpoint_path = str(MODELS_DIR / "vgp_ckpts.{epoch:02d}-{val_loss:.4f}.h5")
        try:
            self.model.load_weights(checkpoint_path)
        except Exception as e:
            print(f"Couldn't load any checkpoints: {e}")

        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, save_best_only=True, save_weights_only=True,
        )
        early_stop_callback = tf.keras.callbacks.EarlyStopping(patience=500)

        self.model.fit(
            self.observation_indices,
            observations,
            batch_size=self.batch_size,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=[checkpoint_callback, early_stop_callback],
        )


if __name__ == "__main__":
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment("/Users/awm1017@ic.ac.uk/VGP SSE Model")

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
    with mlflow.start_run(run_name="laptop_loading_test"):
        mlflow.tensorflow.autolog(1)
        vgp.train_model(train_sses, (index_points, test_sses), epochs=1)
    # vgp.model.save_weights(str(MODELS_DIR / "vgp_v1.h5"))
