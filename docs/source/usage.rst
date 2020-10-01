Usage
=====

UnlockGNN's main functionality comes in two parts: a tool for extracting the penultimate layer outputs from a pretrained GNN
(:class:`~unlockgnn.datalib.preprocessing.LayerExtractor` and :class:`~unlockgnn.datalib.preprocessing.LayerScaler`) and
two types of Gaussian process implementations (see :ref:`Types of Gaussian process`) for learning to perform uncertainty
quantification using the extracted layer outputs for a training data set.

This 

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
