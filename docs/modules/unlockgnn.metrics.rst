.. currentmodule:: unlockgnn.metrics

Uncertainty quantification metrics
==================================

Although the probabilistic models implemented in unlockGNN use
the keras API, currently there is poor support for models
that predict distributions, rather than tensors. In particular,
often the `keras.metrics` API is incompatible. To that end,
unlockGNN provides some alternative utilities for computing metrics,
including some useful uncertainty quantifier-specific metrics.

The main interface for computing metrics is the :func:`evaluate_uq_metrics`
function:

.. autofunction:: evaluate_uq_metrics

.. autodata:: AVAILABLE_METRICS

Uncertainty quantifier-specific metrics
---------------------------------------

.. autofunction:: neg_log_likelihood

.. autofunction:: sharpness

.. autofunction:: variation
