.. currentmodule:: unlockgnn.model

Quickstart for MEGNet users
===========================

UnlockGNN currently contains implementations for adding uncertainty
quantification to a :class:`MEGNetModel` with minimal overhead. After
installing unlockGNN (see :ref:`installation`), you can easily
add uncertainty quantification to a trained :class:`MEGNetModel`.
See also the example scripts on `GitHub <https://github.com/a-ws-m/unlockGNN/tree/master/examples>`_.

This document will demonstrate how to add uncertainty quantification
to MEGNet's pre-trained formation energies model. UnlockGNN's method
for adding this uncertainty quantification is explained in
:ref:`probabilistic models`.
The essential steps to adding uncertainty quantification are:

#. Loading/training a :class:`MEGNetModel`.

#. Initializing a :class:`MEGNetProbModel`.

#. A first run of training using :meth:`MEGNetProbModel.train`.

#. Fine tuning the model: unfreezing its ``"GNN"`` layers, using :meth:`MEGNetProbModel.set_frozen`, then training again.

#. Saving the model using :meth:`MEGNetProbModel.save`.

The model can then be reloaded using :meth:`MEGNetProbModel.load`.

In order to train the uncertainty quantifier,
we will use an example dataset of binary compounds that lie on the
convex hull, which we will download from the Materials Project.
This example is also available in notebook format on unlockGNN's
GitHub page.

Running this example script takes approximately 15
minutes on a desktop computer with an Nvidia GTX 1080 GPU,
not including the time it takes to download the data.

.. literalinclude:: megnet_prob_model_mwe.py
