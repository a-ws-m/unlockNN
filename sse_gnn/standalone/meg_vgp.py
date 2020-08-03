"""MEGNet model employing a VGP as an output."""
from operator import itemgetter
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pymatgen
import tensorflow as tf
import tensorflow_probability as tfp
from megnet.activations import softplus2
from megnet.config import DataType
from megnet.data.crystal import CrystalGraph
from megnet.data.graph import GaussianDistance, StructureGraph
from megnet.layers import GaussianExpansion, MEGNetLayer, Set2Set
from tensorflow.keras.layers import Add, Concatenate, Dense, Dropout, Embedding, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import Sequence
from tensorflow.python.keras.utils import losses_utils


class RBFKernelFn(tf.keras.layers.Layer):
    """A radial basis function implementation that works with keras."""

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
    def kernel(self):
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


class ProbabilisticMEGNetModel:
    """A probabilistic implementation of a `MEGNetModel` with uncertainty quantification."""

    def __init__(
        self,
        gp_num_inducing_points: int,
        nfeat_edge: int = None,
        nfeat_global: int = None,
        nfeat_node: int = None,
        nblocks: int = 3,
        lr: float = 1e-3,
        n1: int = 64,
        n2: int = 32,
        n3: int = 16,
        nvocal: int = 95,
        embedding_dim: int = 16,
        nbvocal: int = None,
        bond_embedding_dim: int = None,
        ngvocal: int = None,
        global_embedding_dim: int = None,
        npass: int = 3,
        ntarget: int = 1,
        act: Callable = softplus2,
        metrics: List[str] = ["mse"],
        l2_coef: float = None,
        dropout: float = None,
        graph_converter: StructureGraph = None,
        optimizer_kwargs: Dict = None,
        dropout_on_predict: bool = False,
        batch_size: int = 32,
        weight_sf: int = int(1e4),
        **kwargs,
    ):
        """Initialize model.

        Args:
            gp_num_inducing_points (int): The number of inducing points for the variable
                Gaussian process.
            nfeat_edge: (int) number of bond features
            nfeat_global: (int) number of state features
            nfeat_node: (int) number of atom features
            nblocks: (int) number of MEGNetLayer blocks
            lr: (float) learning rate
            n1: (int) number of hidden units in layer 1 in MEGNetLayer
            n2: (int) number of hidden units in layer 2 in MEGNetLayer
            n3: (int) number of hidden units in layer 3 in MEGNetLayer
            nvocal: (int) number of total element
            embedding_dim: (int) number of embedding dimension
            nbvocal: (int) number of bond types if bond attributes are types
            bond_embedding_dim: (int) number of bond embedding dimension
            ngvocal: (int) number of global types if global attributes are types
            global_embedding_dim: (int) number of global embedding dimension
            npass: (int) number of recurrent steps in Set2Set layer
            ntarget: (int) number of output targets
            act: (object) activation function
            l2_coef: (float or None) l2 regularization parameter
            metrics: (list) List of Keras metrics to be evaluated by the model during training
                and testing
            dropout: (float) dropout rate
            graph_converter: (object) object that exposes a "convert" method for structure to graph conversion
            optimizer_kwargs (dict): extra keywords for optimizer, for example clipnorm and clipvalue.
            dropout_on_predict (bool): whether to use dropout during prediction.
            batch_size (int): the batch size to be used when training.
            weight_sf (int): The scaling factor for the KL divergence. Performs best when equal to the size
                of the input data.
            kwargs (dict): in the case where bond inputs are pure distances (not the expanded distances nor integers
                for embedding, i.e., nfeat_edge=None and bond_embedding_dim=None),
                kwargs can take additional inputs for expand the distance using Gaussian basis.
                centers (np.ndarray): array for defining the Gaussian expansion centers
                width (float): width for the Gaussian basis

        """
        # Build the MEG Model
        self.model = make_pred_megnet_model(
            gp_num_inducing_points,
            nfeat_edge=nfeat_edge,
            nfeat_global=nfeat_global,
            nfeat_node=nfeat_node,
            nblocks=nblocks,
            n1=n1,
            n2=n2,
            n3=n3,
            nvocal=nvocal,
            embedding_dim=embedding_dim,
            nbvocal=nbvocal,
            bond_embedding_dim=bond_embedding_dim,
            ngvocal=ngvocal,
            global_embedding_dim=global_embedding_dim,
            npass=npass,
            ntarget=ntarget,
            act=act,
            l2_coef=l2_coef,
            dropout=dropout,
            dropout_on_predict=dropout_on_predict,
            **kwargs,
        )

        self.batch_size = batch_size

        # * Compile model
        # Determine KL divergence scaling factor
        kl_weight = np.array(batch_size, np.float64) / weight_sf
        loss = VariationalLoss(kl_weight, name="variational_loss")

        opt_params = {"lr": lr}
        if optimizer_kwargs is not None:
            opt_params.update(optimizer_kwargs)
        self.model.compile(Adam(**opt_params), loss, metrics=metrics)

        if graph_converter is None:
            graph_converter = CrystalGraph(
                cutoff=4, bond_converter=GaussianDistance(np.linspace(0, 5, 100), 0.5)
            )

        self.graph_converter = graph_converter

    def structs_to_input(
        self, structs: List[pymatgen.Structure]
    ) -> List[List[np.ndarray]]:
        """Convert structures to a tensor of input vectors."""
        return list(map(self.graph_converter.get_input, structs))

    def train_from_structs(
        self,
        structs: List[pymatgen.Structure],
        targets: List[np.ndarray],
        val_structs: List[pymatgen.Structure],
        val_targets: List[np.ndarray],
        epochs: int = 1000,
        checkpoint_path: Optional[str] = None,
    ):
        """Train from a list of structures and targets."""
        callbacks = []
        if checkpoint_path:
            try:
                self.model.load_weights(checkpoint_path)
            except Exception as e:
                print(f"Couldn't load checkpoints: {e}")

            checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                checkpoint_path, save_best_only=True, save_weights_only=True
            )
            callbacks.append(checkpoint_callback)

        early_stop_callback = tf.keras.callbacks.EarlyStopping(patience=500)
        callbacks.append(early_stop_callback)

        input_list = self.structs_to_input(structs)
        input_gen = GraphSequence(input_list, targets, self.batch_size)

        val_input_list = self.structs_to_input(val_structs)
        val_gen = GraphSequence(val_input_list, val_targets, self.batch_size)

        steps_per_epoch = len(input_list) // self.batch_size
        validation_steps = len(val_input_list) // self.batch_size

        self.model.fit(
            x=input_gen,
            batch_size=self.batch_size,
            epochs=epochs,
            validation_data=val_gen,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=callbacks,
        )


class GraphSequence(Sequence):
    """Handler for generating data for graph inputs."""

    def __init__(self, graphs, targets, batch_size):
        """Initialize graphs, targets and batch size."""
        self.graphs, self.targets = graphs, targets
        self.batch_size = batch_size

        self.indices = np.arange(len(self.graphs))

        if self.batch_size > len(self.graphs):
            raise ValueError(
                f"Batch size ({self.batch_size}) is larger than "
                f"training data length ({len(self.graphs)})."
            )

    def __len__(self) -> int:
        """Get the Sequence's length."""
        return int(np.ceil(len(self.graphs) / self.batch_size))

    def __getitem__(self, idx) -> Tuple[List[List[np.ndarray]], List[np.ndarray]]:
        """Get a tuple of batches from the Sequence."""
        start_idx = (idx * self.batch_size) % len(self.indices)
        end_idx = (start_idx + self.batch_size) % len(self.indices)

        if end_idx <= start_idx:
            inds = np.concatenate(
                (self.indices[start_idx:], self.indices[:end_idx]), axis=None
            )
        else:
            inds = self.indices[start_idx:end_idx]

        ig = itemgetter(*inds)
        batch_x = list(ig(self.graphs))
        batch_y = list(ig(self.targets))

        return batch_x, batch_y

    def on_epoch_end(self):
        """Shuffle indices."""
        np.random.shuffle(self.indices)


def make_pred_megnet_model(
    gp_num_inducing_points: int,
    nfeat_edge: int = None,
    nfeat_global: int = None,
    nfeat_node: int = None,
    nblocks: int = 3,
    n1: int = 64,
    n2: int = 32,
    n3: int = 16,
    nvocal: int = 95,
    embedding_dim: int = 16,
    nbvocal: int = None,
    bond_embedding_dim: int = None,
    ngvocal: int = None,
    global_embedding_dim: int = None,
    npass: int = 3,
    ntarget: int = 1,
    act: Callable = softplus2,
    l2_coef: float = None,
    dropout: float = None,
    dropout_on_predict: bool = False,
    **kwargs,
) -> Model:
    """Make a MEGNet Model.

    Args:
        gp_num_inducing_points (int): The number of inducing points for the variable
            Gaussian process.
        nfeat_edge: (int) number of bond features
        nfeat_global: (int) number of state features
        nfeat_node: (int) number of atom features
        nblocks: (int) number of MEGNetLayer blocks
        n1: (int) number of hidden units in layer 1 in MEGNetLayer
        n2: (int) number of hidden units in layer 2 in MEGNetLayer
        n3: (int) number of hidden units in layer 3 in MEGNetLayer
        nvocal: (int) number of total element
        embedding_dim: (int) number of embedding dimension
        nbvocal: (int) number of bond types if bond attributes are types
        bond_embedding_dim: (int) number of bond embedding dimension
        ngvocal: (int) number of global types if global attributes are types
        global_embedding_dim: (int) number of global embedding dimension
        npass: (int) number of recurrent steps in Set2Set layer
        ntarget: (int) number of output targets
        act: (object) activation function
        l2_coef: (float or None) l2 regularization parameter
        dropout: (float) dropout rate
        dropout_on_predict (bool): Whether to use dropout during prediction and training
        kwargs (dict): in the case where bond inputs are pure distances (not the expanded
                distances nor integers for embedding, i.e., nfeat_edge=None and bond_embedding_dim=None),
                kwargs can take additional inputs for expand the distance using Gaussian basis.
                centers (np.ndarray): array for defining the Gaussian expansion centers
                width (float): width for the Gaussian basis
    Returns:
        (Model) Keras model, ready to run
    """
    # Get the setting for the training kwarg of Dropout
    dropout_training = True if dropout_on_predict else None

    # atom inputs

    if nfeat_node is None:
        # only z as feature
        x1 = Input(shape=(None,), dtype=DataType.tf_int, name="atom_int_input")
        x1_ = Embedding(nvocal, embedding_dim, name="atom_embedding")(x1)
    else:
        x1 = Input(shape=(None, nfeat_node), name="atom_feature_input")
        x1_ = x1

    # bond inputs
    if nfeat_edge is None:
        if bond_embedding_dim is not None:
            # bond attributes are integers for embedding
            x2 = Input(shape=(None,), dtype=DataType.tf_int, name="bond_int_input")
            x2_ = Embedding(nbvocal, bond_embedding_dim, name="bond_embedding")(x2)
        else:
            # the bond attributes are float distance
            x2 = Input(shape=(None,), dtype=DataType.tf_float, name="bond_float_input")
            centers = kwargs.get("centers", None)
            width = kwargs.get("width", None)
            if centers is None and width is None:
                raise ValueError(
                    "If the bond attributes are single float values, "
                    "we expect the value to be expanded before passing "
                    "to the models. Therefore, `centers` and `width` for "
                    "Gaussian basis expansion are needed"
                )
            x2_ = GaussianExpansion(centers=centers, width=width)(x2)  # type: ignore
    else:
        x2 = Input(shape=(None, nfeat_edge), name="bond_feature_input")
        x2_ = x2

    # state inputs
    if nfeat_global is None:
        if global_embedding_dim is not None:
            # global state inputs are embedding integers
            x3 = Input(shape=(None,), dtype=DataType.tf_int, name="state_int_input")
            x3_ = Embedding(ngvocal, global_embedding_dim, name="state_embedding")(x3)
        else:
            # take default vector of two zeros
            x3 = Input(
                shape=(None, 2), dtype=DataType.tf_float, name="state_default_input"
            )
            x3_ = x3
    else:
        x3 = Input(shape=(None, nfeat_global), name="state_feature_input")
        x3_ = x3
    x4 = Input(shape=(None,), dtype=DataType.tf_int, name="bond_index_1_input")
    x5 = Input(shape=(None,), dtype=DataType.tf_int, name="bond_index_2_input")
    x6 = Input(shape=(None,), dtype=DataType.tf_int, name="atom_graph_index_input")
    x7 = Input(shape=(None,), dtype=DataType.tf_int, name="bond_graph_index_input")

    if l2_coef is not None:
        reg = l2(l2_coef)
    else:
        reg = None

    # two feedforward layers
    def ff(x, n_hiddens=[n1, n2], name_prefix=None):
        if name_prefix is None:
            name_prefix = "FF"
        out = x
        for k, i in enumerate(n_hiddens):
            out = Dense(
                i,
                activation=act,
                kernel_regularizer=reg,
                name="%s_%d" % (name_prefix, k),
            )(out)
        return out

    # a block corresponds to two feedforward layers + one MEGNetLayer layer
    # Note the first block does not contain the feedforward layer since
    # it will be explicitly added before the block
    def one_block(a, b, c, has_ff=True, block_index=0):
        if has_ff:
            x1_ = ff(a, name_prefix="block_%d_atom_ff" % block_index)
            x2_ = ff(b, name_prefix="block_%d_bond_ff" % block_index)
            x3_ = ff(c, name_prefix="block_%d_state_ff" % block_index)
        else:
            x1_ = a
            x2_ = b
            x3_ = c
        out = MEGNetLayer(
            [n1, n1, n2],
            [n1, n1, n2],
            [n1, n1, n2],
            pool_method="mean",
            activation=act,
            kernel_regularizer=reg,
            name="megnet_%d" % block_index,
        )([x1_, x2_, x3_, x4, x5, x6, x7])

        x1_temp = out[0]
        x2_temp = out[1]
        x3_temp = out[2]
        if dropout:
            x1_temp = Dropout(dropout, name="dropout_atom_%d" % block_index)(
                x1_temp, training=dropout_training
            )
            x2_temp = Dropout(dropout, name="dropout_bond_%d" % block_index)(
                x2_temp, training=dropout_training
            )
            x3_temp = Dropout(dropout, name="dropout_state_%d" % block_index)(
                x3_temp, training=dropout_training
            )
        return x1_temp, x2_temp, x3_temp

    x1_ = ff(x1_, name_prefix="preblock_atom")
    x2_ = ff(x2_, name_prefix="preblock_bond")
    x3_ = ff(x3_, name_prefix="preblock_state")
    for i in range(nblocks):
        if i == 0:
            has_ff = False
        else:
            has_ff = True
        x1_1 = x1_
        x2_1 = x2_
        x3_1 = x3_
        x1_1, x2_1, x3_1 = one_block(x1_1, x2_1, x3_1, has_ff, block_index=i)
        # skip connection
        x1_ = Add(name="block_%d_add_atom" % i)([x1_, x1_1])
        x2_ = Add(name="block_%d_add_bond" % i)([x2_, x2_1])
        x3_ = Add(name="block_%d_add_state" % i)([x3_, x3_1])

    # print(Set2Set(T=npass, n_hidden=n3, kernel_regularizer=reg, name='set2set_atom'
    #             ).compute_output_shape([i.shape for i in [x1_, x6]]))
    # set2set for both the atom and bond
    node_vec = Set2Set(
        T=npass, n_hidden=n3, kernel_regularizer=reg, name="set2set_atom"
    )([x1_, x6])
    # print('Node vec', node_vec)
    edge_vec = Set2Set(
        T=npass, n_hidden=n3, kernel_regularizer=reg, name="set2set_bond"
    )([x2_, x7])
    # concatenate atom, bond, and global
    final_vec = Concatenate(axis=-1)([node_vec, edge_vec, x3_])
    if dropout:
        final_vec = Dropout(dropout, name="dropout_final")(
            final_vec, training=dropout_training
        )
    # final dense layers
    final_vec = Dense(
        n2, activation=act, kernel_regularizer=reg, dtype=tf.float64, name="readout_0"
    )(final_vec)

    # * GP layer
    kernel = RBFKernelFn(dtype=tf.float64)
    out = tfp.layers.VariationalGaussianProcess(
        gp_num_inducing_points,
        kernel,
        event_shape=(ntarget,),
        convert_to_tensor_fn=tfp.distributions.Distribution.mean,
        name="variational_gp",
    )(final_vec)
    model = Model(inputs=[x1, x2, x3, x4, x5, x6, x7], outputs=out)

    return model
