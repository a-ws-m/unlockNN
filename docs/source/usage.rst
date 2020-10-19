Usage
=====

UnlockGNN's main functionality comes in two parts: a tool for extracting the penultimate layer outputs from a pretrained GNN
(:class:`~unlockgnn.datalib.preprocessing.LayerExtractor` and :class:`~unlockgnn.datalib.preprocessing.LayerScaler`) and
two types of Gaussian process implementations (see :ref:`Types of Gaussian process`) for learning to perform uncertainty
quantification using the extracted layer outputs for a training data set.

On top of these back end tools is the |probgnn| abstract base class, which provides a single interface
for adding an uncertainty quantifier to graph neural networks.
It is discussed in :ref:`Implementing |probgnn| for a different GNN`.
It has been implemented for ``MEGNetModel``\ s: see :class:`~unlockgnn.MEGNetProbModel`.

Types of Gaussian process
-------------------------

UnlockGNN provides two types of uncertainty quantifier based on Gaussian processes.
The key differences between each of the techniques are the cost function, the kernel function and the data type of each model:

Variational Gaussian Process (VGP)
    :class:`~unlockgnn.gp.vgp_trainer.SingleLayerVGP` makes use of the `VariationalGaussianProcess <https://www.tensorflow.org/probability/api_docs/python/tfp/layers/VariationalGaussianProcess>`_
    Keras layer and uses a `variational loss function <https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/VariationalGaussianProcess#variational_loss>`_.
    It uses a radial basis function kernel, defined in :class:`~unlockgnn.gp.vgp_trainer.RBFKernelFn`.
Gaussian Process (GP)
    :class:`~unlockgnn.gp.gp_trainer.GPTrainer` inherits from `tf.Module <https://www.tensorflow.org/api_docs/python/tf/Module>`_
    and uses negative log likelihood loss and a `Matern kernel with parameter 1/2 <https://www.tensorflow.org/api_docs/python/tf/Module>`_.

Implementing |probgnn| for a different GNN
------------------------------------------

|probgnn| handles Gaussian process instantiation and training, as well as model and data saving and loading.
It also implements a :meth:`~unlockgnn.ProbGNN.predict` method for obtaining a result and an uncertainty without
having to manage the data pipeline between the GNN and the uncertainty quantifier.

There are three ``abstractmethod``\ s that need to be implemented in a class that inherits from |probgnn|:

* :meth:`~unlockgnn.ProbGNN.make_gnn` must construct and return an untrained GNN model, which is assigned
    to the :attr:`~unlockgnn.ProbGNN.gnn` attribute.
* :meth:`~unlockgnn.ProbGNN.train_gnn` must train the GNN and save it to :attr:`~unlockgnn.ProbGNN.gnn_save_path`.
    :attr:`~unlockgnn.ProbGNN.gnn_ckpt_path` should be used for saving checkpoints, if applicable.
* :meth:`~unlockgnn.ProbGNN.load_gnn` must load a pre-trained GNN from the disk (from :attr:`~unlockgnn.ProbGNN.gnn_save_path`).

For an example implementation, see the :class:`~unlockgnn.MEGNetProbModel` class.

.. |probgnn| replace:: :class:`~unlockgnn.ProbGNN`
