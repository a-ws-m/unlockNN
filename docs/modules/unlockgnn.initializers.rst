.. currentmodule:: unlockgnn.initializers

Inducing index points initializers
==================================

The ``initializers`` module contains work-in-progress algorithms for more intelligently
initializing the locations of inducing index points
for the variational Gaussian process (VGP).
As per the `TensorFlow documentation <https://www.tensorflow.org/probability/api_docs/python/tfp/layers/VariationalGaussianProcess#args>`_,
VGPs are highly sensitive to the location of these points, so improving the initializer could substantially increase performance.

So far, only one custom initializer has been implemented and it does not seem to offer any performance improvement
over the default initializer:

.. autoclass:: SampleInitializer
