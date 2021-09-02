.. currentmodule:: unlocknn.kernel_layers

Kernel layers
=============

UnlockNN provides Keras-compatible implementations of a few kernel
functions for use with the Gaussian process, as well as an extensible
API for implementing your own kernels.

Built-in kernels
----------------

.. autosummary::

   RBFKernelFn
   MaternOneHalfFn

Each of these kernels can be saved using their :meth:`KernelLayer.save` method and then
loaded using :func:`load_kernel`:

.. autofunction:: load_kernel

Extending the kernels API
-------------------------

All kernels should:

* Inherit from the :class:`KernelLayer` abstract base class
* Implement the :meth:`KernelLayer.kernel` property
* Add appropriate "weights" (kernel parameters) during :meth:`KernelLayer.__init__`
* Implement :meth:`KernelLayer.config` for compatibility with :func:`load_kernel`

.. autoclass:: KernelLayer
   :members:
   :special-members: __init__

Both of the current built-in kernels are parameterized by an amplitude and
a length scale. The :class:`AmpAndLengthScaleFn` abstract base class initializes
these weights during instantiation and other kernels that are parameterized as
such may inherit from this class also:

.. autoclass:: AmpAndLengthScaleFn
   :special-members: __init__
