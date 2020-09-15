"""VariationalGaussianProcess single layer model."""
from typing import List, Optional, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.callbacks import Callback
from tensorflow.python.keras.utils import losses_utils


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
        num_inducing_points (int): The number of inducing points for the :obj:`VariationalGaussianProcess`.
        ntargets (int): The number of parameters to be modelled.
        batch_size (int): The training batch size.
        prev_model (str, optional): The path to a previously saved model.

    Attributes:
        observation_indices (:obj:`tf.Tensor`): The (training) observation index points (x data).
        batch_size (int): The training batch size.
        model (:obj:`Model`): The Keras model containing the obj:`VariationalGaussianProcess`.
        loaded_modl (bool): Whether a previously trained model was loaded.

    """

    def __init__(
        self,
        observation_indices: tf.Tensor,
        num_inducing_points: int,
        ntargets: int = 1,
        batch_size: int = 32,
        prev_model: Optional[str] = None,
    ):
        """Initialize and compile model."""
        self.observation_indices = observation_indices
        self.batch_size = batch_size

        # * Set up model
        input_shape = observation_indices.shape[1:]
        inputs = tf.keras.layers.Input(shape=input_shape)
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

        self.loaded_model = False
        if prev_model:
            self.model.load_weights(prev_model)
            self.loaded_model = True

    def __call__(self, *args, **kwargs) -> tfp.distributions.Distribution:
        """Call the embedded Keras model."""
        return self.model.call(*args, **kwargs).distribution

    def predict(self, index_points: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Predict target value and uncertainty for index points."""
        dist = self(index_points)
        return dist.mean(), dist.stddev()

    def train_model(
        self,
        observations: tf.Tensor,
        validation_data: Optional[Tuple] = None,
        epochs: int = 1000,
        checkpoint_path: Optional[str] = None,
        patience: int = 500,
        callbacks: List[Callback] = [],
    ):
        """Train the model.

        Args:
            observations (:obj:`tf.Tensor`): The observed true `y` values.
            validation_data (tuple of :obj:`tf.Tensor`, optional): The validation data
                as a tuple of ``(validation_x, validation_y)``.
            epochs (int): The number of training epochs.
            checkpoint_path (str, optional): The path to save new checkpoints to.
                If :attr:`loaded_model` is `False`, will try to load checkpoints from
                this path as well.
            patience (int): The number of iterations to continue training without
                validation loss improvement before stopping training early.
            callbacks (list of :obj:`Callback`): A list of additional callbacks.

        """
        if checkpoint_path:
            if not self.loaded_model:
                try:
                    self.model.load_weights(checkpoint_path)
                except Exception as e:
                    print(f"Couldn't load any checkpoints: {e}")

            checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                checkpoint_path, save_best_only=True, save_weights_only=True,
            )
            callbacks.append(checkpoint_callback)

        early_stop_callback = tf.keras.callbacks.EarlyStopping(patience=patience)
        callbacks.append(early_stop_callback)

        self.model.fit(
            self.observation_indices,
            observations,
            batch_size=self.batch_size,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=callbacks,
        )
