Usage
=====

Types of Gaussian process
-------------------------

UnlockGNN provides two types of uncertainty quantifier based on Gaussian processes.
The key differences between each of the techniques are the cost function, the kernel function and the data type of each model:

Variational Gaussian Process (VGP)
    makes use of the `VariationalGaussianProcess <https://www.tensorflow.org/probability/api_docs/python/tfp/layers/VariationalGaussianProcess>`_
    Keras layer and uses a `variational loss function <https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/VariationalGaussianProcess#variational_loss>`_.
    It uses a radial basis function kernel.
Gaussian Process (GP)
    inherits from `tf.Module <https://www.tensorflow.org/api_docs/python/tf/Module>`_
    and uses negative log likelihood loss and a `Matern kernel with parameter 1/2 <https://www.tensorflow.org/api_docs/python/tf/Module>`_.
