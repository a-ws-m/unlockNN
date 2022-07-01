"""ProbNN model code and implementation for MEGNet."""
import json
from shutil import copyfile
from os import PathLike
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from typing import Literal, get_args
except ImportError:  # pragma: no cover
    from typish import Literal, get_args


import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
from megnet.models import MEGNetModel
from megnet.utils.preprocessing import DummyScaler
from pymatgen.core import Structure
from tensorflow.python.keras.utils import losses_utils

from .kernel_layers import KernelLayer, RBFKernelFn, load_kernel
from .megnet_utils import create_megnet_input, Targets, ModelInput

tfd = tfp.distributions

LayerName = Literal["NN", "VGP", "Norm"]
Metrics = List[Union[str, tf.keras.metrics.Metric]]

__all__ = ["ProbNN", "MEGNetProbModel"]


def _get_save_paths(root_dir: PathLike) -> Dict[str, Path]:
    """Get default save paths for model components.

    Args:
        save_dir: The root save directory.

    Returns:
        Dictionary of ``{component: path}``.

    """
    root_dir = Path(root_dir)
    rel_paths: Dict[str, str] = dict(
        weights_path="weights",
        conf_path="config.json",
        ckpt_path="checkpoint.h5",
        kernel_path="kernel",
        nn_path="nn",
        meg_path="megnet",
    )
    return {component: root_dir / rel_path for component, rel_path in rel_paths.items()}


class VariationalLoss(keras.losses.Loss):
    """Implementation of variational loss using keras API."""

    def __init__(
        self,
        kl_weight: float,
        reduction: keras.losses.Reduction = losses_utils.ReductionV2.AUTO,
        name: str = "variational_loss",
    ):
        """Initialize loss function and KL divergence loss scaling factor."""
        self.kl_weight = kl_weight
        super().__init__(reduction=reduction, name=name)

    def call(self, y_true, predicted_distribution):
        """Calculate the variational loss."""
        return predicted_distribution.variational_loss(y_true, kl_weight=self.kl_weight)


def make_probabilistic(
    nn: keras.Model,
    num_inducing_points: int,
    kernel: KernelLayer = RBFKernelFn(),
    latent_layer: Union[str, int] = -2,
    target_shape: Union[Tuple[int], int] = 1,
    prediction_mode: bool = False,
    index_initializer: Optional[keras.initializers.Initializer] = None,
    use_normalization: bool = False,
) -> keras.Model:
    """Make a neural network probabilistic by replacing the final layer(s) with a VGP.

    Caution: This function modifies the neural network (NN) in memory.
    Ensure that the NN has been saved to disk before using.

    Args:
        nn: The base NN model to modify. latent_layer: The name or index of
            the layer of the NN to be fed into the VGP.
        num_inducing_points: The number of inducing index points for the
            VGP.
        kernel: A :class:`KernelLayer` for the VGP to use.
        latent_layer: The index or name of the NN layer to use as the
            input for the VGP.
        target_shape: The shape of the target values.
        use_normalization: Whether to use a ``BatchNormalization`` layer before
            the VGP. Recommended for better training efficiency.
        prediction_mode: Whether to create a model for predictions _only_.
            (Resulting model cannot be serialized and loss functions won't work.)
        index_initializer: A custom initializer to use for the VGP index points.

    Returns:
        A :class:`keras.Model` with the ``nn``'s first layers, but terminating in a
            VGP.

    """
    if isinstance(latent_layer, int):
        latent_idx = latent_layer
    else:
        latent_idx = [layer.name for layer in nn.layers].index(latent_layer)

    vgp_input = nn.layers[latent_idx].output

    if use_normalization:
        vgp_input = keras.layers.BatchNormalization()(vgp_input)
        if index_initializer is None:
            index_initializer = keras.initializers.TruncatedNormal(stddev=1.0)

    output_shape = (
        (target_shape,) if isinstance(target_shape, int) else tuple(target_shape)
    )
    convert_fn = (
        "mean"
        if not prediction_mode
        else lambda trans_dist: tf.concat(
            [trans_dist.distribution.mean(), trans_dist.distribution.stddev()], axis=-1
        )
    )
    vgp_outputs = tfp.layers.VariationalGaussianProcess(
        num_inducing_points,
        kernel,
        event_shape=output_shape,
        inducing_index_points_initializer=index_initializer,
        convert_to_tensor_fn=convert_fn,
    )(vgp_input)

    return keras.Model(nn.inputs, vgp_outputs)


class ProbNN(ABC):
    """Wrapper for creating a probabilistic NN model.

    Args:
        nn: The base NN model to modify.
        num_inducing_points: The number of inducing index points for the
            VGP.
        kernel: A :class:`~unlocknn.kernel_layers.KernelLayer` for the VGP to use.
        latent_layer: The name or index of the layer of the NN to be fed into
            the VGP.
        target_shape: The shape of the target values.
        metrics: A list of metrics to record during training.
        kl_weight: The relative weighting of the Kullback-Leibler divergence
            in the loss function.
        optimizer: The model optimizer, needed for recompilation.
        index_initializer: A custom initializer to use for the VGP index points.
        use_normalization: Whether to use a ``BatchNormalization`` layer before
            the VGP. Recommended for better training efficiency.
        compile: Whether to compile the model for training. Not needed when loading
            the model for inference only.

    Attributes:
        CONFIG_VARS: A list of attribute names, as strings, to include in metadata
            when saving. These variables will be saved in a ``config.json`` file
            and used when re-instantiating the model upon loading with
            :meth:`load`: they are passed as keyword arguments.

    """

    CONFIG_VARS: List[str] = [
        "num_inducing_points",
        "metrics",
        "kl_weight",
        "latent_layer",
        "target_shape",
        "use_normalization",
    ]

    def __init__(
        self,
        nn: keras.Model,
        num_inducing_points: int,
        kernel: KernelLayer = RBFKernelFn(),
        latent_layer: Union[str, int] = -2,
        target_shape: Union[Tuple[int], int] = 1,
        metrics: Metrics = [],
        kl_weight: float = 1.0,
        optimizer: keras.optimizers.Optimizer = tf.optimizers.Adam(),
        index_initializer: Optional[keras.initializers.Initializer] = None,
        use_normalization: bool = True,
    ) -> None:
        """Initialize the probabilistic model.

        The model's NN layers are initially frozen by default (but not the VGP
        and ``BatchNormalization`` layer, if applicable).

        """
        self.nn = nn
        self.num_inducing_points = num_inducing_points
        self.kernel = kernel
        self.latent_layer = latent_layer
        self.target_shape = target_shape
        self.metrics = metrics
        self.kl_weight = kl_weight
        self.optimizer: keras.optimizers.Optimizer = optimizer
        self.use_normalization = use_normalization
        self.pred_model: Optional[keras.Model] = None

        # Instantiate probabilistic model
        self.model = make_probabilistic(
            nn,
            num_inducing_points,
            kernel,
            latent_layer,
            target_shape,
            index_initializer=index_initializer,
            use_normalization=use_normalization,
        )

        # Freeze NN layers and compile, ready to train the VGP
        self.set_frozen("NN", recompile=True)

    def set_frozen(
        self,
        layers: Union[LayerName, List[LayerName]],
        freeze: bool = True,
        recompile: bool = True,
        **compilation_kwargs,
    ) -> None:
        """Freeze or thaw probabilistic NN layers.

        Args:
            layers: Name or list of names of layers to thaw.
            freeze: Whether to freeze (``True``) or thaw (``False``) the layers.
            recompile: Whether to recompile the model after the operation.
            **compilation_kwargs: Keyword arguments to pass to :meth:`compile`.

        Raises:
            ValueError: If one or more ``layers`` are invalid names.

        """
        if not isinstance(layers, list):
            layers = [layers]

        # Validation check
        valid_layers = get_args(LayerName)
        for name in layers:
            if name not in valid_layers:
                raise ValueError(
                    f"Cannot freeze `{name}`; must be one of {valid_layers}."
                )

        if "Norm" in layers:
            if not self.use_normalization:
                raise ValueError(
                    "Cannot freeze normalization layer: `use_normalization` is False."
                )
            else:
                self.model.layers[-2].trainable = not freeze
        if "NN" in layers:
            last_nn_index = -2 if self.use_normalization else -1
            for layer in self.model.layers[:last_nn_index]:
                layer.trainable = not freeze
        if "VGP" in layers:
            self.model.layers[-1].trainable = not freeze

        if recompile:
            self.compile(optimizer=self.optimizer, **compilation_kwargs)

    @abstractmethod
    def train(self, *args, **kwargs) -> None:
        """Train the model.

        This method should handle data processing, then call ``self.model.fit``
        to train the underlying model.

        """
        ...

    def _update_pred_model(self) -> None:
        """Handle updating predictor model once checks have been done."""
        weights = self.model.get_weights()
        pred_model = make_probabilistic(
            self.nn,
            self.num_inducing_points,
            self.kernel,
            self.latent_layer,
            self.target_shape,
            use_normalization=self.use_normalization,
            prediction_mode=True,
        )
        pred_model.set_weights(weights)
        self.pred_model = pred_model

    def update_pred_model(self, force_new: bool = False) -> None:
        """Instantiate or update the predictor model.

        The predictor model is saved in :attr:`pred_model`. This method is a
        workaround to reconcile the inability to save or train a model that
        returns the VGP distribution's mean *and* standard deviation
        simultaneously.

        This method creates a clone model and so it must be called before making
        a prediction whenever the :attr:`model`'s weights have changed. By
        default, the method checks whether the pre-existing :attr:`pred_model`'s
        weights are similar to the :attr:`model`'s weights before cloning, and
        skips execution if they are. Setting ``force_new=True`` skips this check.

        Args:
            force_new: Whether to force the creation of a new model, skipping
                the weights equality check.

        """
        if force_new or self.pred_model is None:
            self._update_pred_model()
        else:
            # Check if we need to instantiate the prediction model
            current_weights = self.model.get_weights()
            predictor_weights = self.pred_model.get_weights()
            needs_update = not all(
                np.allclose(current, predictor, equal_nan=True)
                for current, predictor in zip(current_weights, predictor_weights)
            )
            if needs_update:
                self._update_pred_model()

    def predict(self, input) -> np.ndarray:
        """Predict target values and standard deviations for a given input.

        Args:
            input: The input(s) to the model.

        Returns:
            A numpy array containing predicted means and standard deviations.

        """
        self.update_pred_model()
        return self.pred_model.predict(input)

    @property
    def nn_frozen(self) -> bool:
        """Determine whether all NN layers are frozen."""
        last_nn_index = -2 if self.use_normalization else -1
        return all((not layer.trainable for layer in self.model.layers[:last_nn_index]))

    @property
    def vgp_frozen(self) -> bool:
        """Determine whether the VGP is frozen."""
        return not self.model.layers[-1].trainable

    @property
    def norm_frozen(self) -> Optional[bool]:
        """Determine whether the BatchNormalization layer is frozen, if it exists."""
        if self.use_normalization:
            return not self.model.layers[-2].trainable

    def compile(
        self,
        new_kl_weight: Optional[float] = None,
        optimizer: keras.optimizers.Optimizer = tf.optimizers.Adam(),
        new_metrics: Optional[Metrics] = None,
    ):
        """Compile the probabilistic NN.

        Recompilation is required whenever layers are (un)frozen.

        Args:
            new_kl_weight: The relative weighting of the Kullback-Leibler divergence
                in the loss function. Default (``None``) is to leave unchanged.
            optimizer: The model optimizer, needed for recompilation.
            new_metrics: New metrics with which to compile.

        """
        loss = VariationalLoss(
            new_kl_weight if new_kl_weight is not None else self.kl_weight
        )
        if new_metrics is not None:
            self.metrics = new_metrics
        self.model.compile(optimizer, loss=loss, metrics=self.metrics)

    def ckpt_callback(self, ckpt_path: PathLike = "checkpoint.h5"):
        """Get the default configuration for a model checkpoint callback."""
        return keras.callbacks.ModelCheckpoint(
            ckpt_path, save_best_only=True, save_weights_only=True
        )

    @property
    def config(self) -> Dict[str, Any]:
        """Get the configuration parameters needed to save to disk."""
        return {var_name: getattr(self, var_name) for var_name in self.CONFIG_VARS}

    def save(
        self, save_path: PathLike, ckpt_path: Optional[PathLike] = "checkpoint.h5"
    ):
        """Save the model to disk.

        Args:
            save_path: The directory in which to save the model.
            ckpt_path: Where to look for checkpoints, which will be
                copied over to the save directory for future usage.
                Specify ``ckpt_path=None`` if no checkpoints exist.

        """
        paths = _get_save_paths(save_path)

        self.model.save_weights(paths["weights_path"])

        with paths["conf_path"].open("w") as f:
            json.dump(self.config, f)

        self.save_kernel(paths["kernel_path"])

        self.nn.save(paths["nn_path"], include_optimizer=False)

        # Copy checkpoints
        if ckpt_path is not None:
            ckpt_path = Path(ckpt_path)
            copyfile(ckpt_path, paths["ckpt_path"])

    def save_kernel(self, kernel_save_path: Path):
        """Save the VGP's kernel to disk."""
        self.kernel.save(kernel_save_path)

    @classmethod
    def load(
        cls: "ProbNN", save_path: PathLike, load_ckpt: bool = True, **kwargs
    ) -> "ProbNN":
        """Load a ProbNN from disk.

        Loaded models must be recompiled before training.

        Args:
            save_path: The path to the model's save directory.
            load_ckpt: Whether to load the best checkpoint's weights, instead
                of those saved at the time of the last :meth:`save`.
            **kwargs: Keyword arguments required by subclasses.

        Returns:
            The loaded model.

        Raises:
            FileNotFoundError: If the ``save_path`` or any components do not exist.

        """
        # Check the save path exists
        if not Path(save_path).exists():
            raise FileNotFoundError(f"`{save_path}` does not exist.")

        paths = _get_save_paths(save_path)

        # Load configuration info
        with paths["conf_path"].open("r") as f:
            config: Dict[str, Any] = json.load(f)

        # Load NN
        try:
            nn = keras.models.load_model(paths["nn_path"], compile=False)
        except OSError:
            raise FileNotFoundError(f"Couldn't load NN at `{paths['nn_path']}`.")

        # Initialize...
        config.update(kwargs)
        prob_model: ProbNN = cls(nn=nn, **config)

        # ...and load weights
        to_load = paths["ckpt_path"] if load_ckpt else paths["weights_path"]
        try:
            prob_model.model.load_weights(to_load)
        except OSError:
            warnings.warn(f"No saved weights found at `{to_load}`.")

        return prob_model


class MEGNetProbModel(ProbNN):
    """:class:`ProbNN` for MEGNetModels.

    Args:
        meg_model: The base :class:`MEGNetModel` to modify.
        num_inducing_points: The number of inducing index points for the
            VGP.
        kernel: A :class:`~unlocknn.kernel_layers.KernelLayer` for the VGP to use.
        latent_layer: The name or index of the layer of the NN to be fed into
            the VGP.
        target_shape: The shape of the target values.
        metrics: A list of metrics to record during training.
        kl_weight: The relative weighting of the Kullback-Leibler divergence
            in the loss function.
        optimizer: The model optimizer, needed for recompilation.
        index_initializer: A custom initializer to use for the VGP index points.
            See also :mod:`unlocknn.initializers`.
        use_normalization: Whether to use a ``BatchNormalization`` layer before
            the VGP. Recommended for better training efficiency.

    .. warning::
        :attr:`metrics` are malfunctional and may give vastly incorrect
        values -- use :func:`unlocknn.metrics.evaluate_uq_metrics` instead!

    """

    def __init__(
        self,
        meg_model: MEGNetModel,
        num_inducing_points: int,
        kernel: KernelLayer = RBFKernelFn(),
        latent_layer: Union[str, int] = -2,
        target_shape: Optional[Union[Tuple[int], int]] = None,
        metrics: List[Union[str, tf.keras.metrics.Metric]] = [],
        kl_weight: float = 1.0,
        optimizer: keras.optimizers.Optimizer = tf.optimizers.Adam(),
        index_initializer: Optional[keras.initializers.Initializer] = None,
        use_normalization: bool = True,
        **kwargs,
    ) -> None:
        """Initialize probabilistic model."""
        self.meg_model = meg_model
        if target_shape is None:
            # Determine output shape based on MEGNetModel
            target_shape = self.meg_model.model.layers[-1].output_shape
            target_shape = tuple(dim for dim in target_shape if dim is not None)

        super().__init__(
            self.meg_model.model,
            num_inducing_points,
            kernel,
            latent_layer,
            target_shape,
            metrics,
            kl_weight,
            optimizer,
            index_initializer,
            use_normalization,
        )

    def train(
        self,
        inputs: List[Structure],
        targets: Targets,
        epochs: int,
        val_inputs: Optional[List[ModelInput]] = None,
        val_targets: Optional[Targets] = None,
        callbacks: List[tf.keras.callbacks.Callback] = [],
        use_default_ckpt_handler: bool = True,
        ckpt_path: PathLike = "checkpoint.h5",
        batch_size: int = 32,
        scrub_failed_structs: bool = False,
        verbose: Literal[0, 1, 2] = 2,
    ):
        """Train the model.

        Args:
            inputs: A list of training crystal structures or graphs.
            targets: A list of training target values.
            epochs: The number of training epochs.
            val_inputs: A list of validation crystal structures or graphs.
            val_targets: A list of validation target values.
            callbacks: A list of additional callbacks.
            use_default_ckpt_handler: Whether to use the default
                checkpoint callback.
            ckpt_path: Where to save checkpoints, if
                ``use_default_ckpt_handler=True``.
            batch_size: The batch size for training and validation.
            scrub_failed_structures: Whether to discard structures
                that could not be converted to graphs.
            verbose: The level of verbosity. See Keras's documentation
                on ``Model.fit``.

        """
        # Convert structures to graphs for model input
        train_gen, train_graphs = create_megnet_input(
            self.meg_model, inputs, targets, batch_size, scrub_failed_structs
        )
        steps_per_train = int(np.ceil(len(train_graphs) / batch_size))

        val_gen = None
        val_graphs = None
        steps_per_val = None
        if val_inputs is not None and val_targets is not None:
            val_gen, val_graphs = create_megnet_input(
                self.meg_model,
                val_inputs,
                val_targets,
                batch_size,
                scrub_failed_structs,
            )
            steps_per_val = int(np.ceil(len(val_graphs) / batch_size))

        # Configure callbacks
        if use_default_ckpt_handler:
            callbacks.append(self.ckpt_callback(ckpt_path))

        # Train
        self.model.fit(
            train_gen,
            steps_per_epoch=steps_per_train,
            validation_data=val_gen,
            validation_steps=steps_per_val,
            epochs=epochs,
            verbose=verbose,
            callbacks=callbacks,
        )

    def evaluate(
        self,
        eval_structs: List[Structure],
        eval_targets: Targets,
        batch_size: int = 128,
        scrub_failed_structs: bool = False,
    ) -> Dict[str, float]:
        """Evaluate model metrics.

        .. warning::
            This method is malfunctional and may give vastly incorrect
            values -- use :func:`unlocknn.metrics.evaluate_uq_metrics` instead!

        Args:
            eval_structs: Structures on which to evaluate performance.
            eval_targets: True target values for structures.
            batch_size: The batch size for training and validation.
            scrub_failed_structures: Whether to discard structures
                that could not be converted to graphs.

        Returns:
            Dictionary of ``{metric: value}``.

        """
        eval_gen, eval_graphs = create_megnet_input(
            self.meg_model, eval_structs, eval_targets, batch_size, scrub_failed_structs
        )
        steps = int(np.ceil(len(eval_graphs) / batch_size))
        return self.model.evaluate(
            eval_gen, batch_size=batch_size, steps=steps, return_dict=True
        )

    def predict(
        self, input: Union[ModelInput, List[ModelInput]], batch_size: int = 128
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict target values and standard deviations for a given input.

        Args:
            input: The input structure(s).
            batch_size: The batch size for predictions.

        Returns:
            The mean values of the predicted distribution(s).

            The standard deviations of the predicted distribution(s).

        Examples:
            Predict the formation energy of a binary compound with a
            95% confidence interval (two standard deviations) uncertainty
            estimate:

            >>> from unlocknn.download import load_data, load_pretrained
            >>> binary_model = load_pretrained("binary_e_form")
            >>> binary_data = load_data("binary_e_form")
            >>> example_struct = binary_data.loc[0, "structure"]
            >>> prediction, stddev = binary_model.predict(example_struct)
            >>> print(
            ...     "Predicted formation energy = "
            ...     f"{prediction.item():.3f} ± {stddev.item() * 2:.3f} eV."
            ... )
            Predicted formation energy = -0.736 ± 0.054 eV.

        """
        self.update_pred_model()

        try:
            len(input)
        except TypeError:
            # Make it a sequence
            input = [input]

        inputs, graphs = create_megnet_input(
            self.meg_model, inputs=input, batch_size=batch_size, shuffle=False
        )
        num_atoms = [len(graph["atom"]) for graph in graphs]

        means = []
        stddevs = []
        for inp in inputs:
            prediction = self.pred_model.predict(inp[:-1]).squeeze()
            n_batch = int(len(prediction) / 2)
            means.append(prediction[:n_batch])
            stddevs.append(prediction[n_batch:])

        means = np.concatenate(means)
        stddevs = np.concatenate(stddevs)

        if not isinstance(self.meg_model.target_scaler, DummyScaler):
            means = self.meg_model.target_scaler.inverse_transform(means, num_atoms)
            stddevs = self.meg_model.target_scaler.inverse_transform(stddevs, num_atoms)

        return means, stddevs

    def save(
        self, save_path: PathLike, ckpt_path: Optional[PathLike] = "checkpoint.h5"
    ) -> None:
        """Save the model to disk.

        Args:
            save_path: The directory in which to save the model.
            ckpt_path: Where to look for checkpoints, which will be
                copied over to the save directory for future usage.
                Specify ``ckpt_path=None`` if no checkpoints exist.

        """
        paths = _get_save_paths(save_path)
        self.meg_model.save_model(str(paths["meg_path"]))
        return super().save(save_path, ckpt_path=ckpt_path)

    @classmethod
    def load(
        cls: "MEGNetProbModel", save_path: PathLike, load_ckpt: bool = True
    ) -> "MEGNetProbModel":
        """Load a MEGNetProbModel from disk.

        Args:
            save_path: The path to the model's save directory.
            load_ckpt: Whether to load the best checkpoint's weights, instead
                of those saved at the time of the last :meth:`save`.

        Returns:
            The loaded model.

        Raises:
            FileNotFoundError: If the ``save_path`` or any components do not exist.

        """
        paths = _get_save_paths(save_path)
        try:
            meg_model = MEGNetModel.from_file(str(paths["meg_path"]))
        except OSError:
            raise FileNotFoundError(f"No saved MEGNetModel at `{paths['meg_path']}`.")
        return super().load(save_path, load_ckpt, meg_model=meg_model)


if __name__ == "__main__":  # pragma: no cover
    import doctest

    doctest.testmod()
