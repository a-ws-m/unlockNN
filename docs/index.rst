.. unlockGNN documentation master file, created by
   sphinx-quickstart on Wed Aug 18 15:33:30 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
.. module:: unlockgnn

What is unlockGNN?
==================

UnlockGNN is a Python package for adding uncertainty quantification to
Keras models for materials property prediction.

This documentation was built on |today|.

Installation
------------

Using ``pip``::

   $ pip install unlockGNN

From source
^^^^^^^^^^^

First, clone the repository::

   $ git clone https://github.com/a-ws-m/unlockGNN.git
   $ cd unlockGNN

Then, either install dependencies using ``conda``::

   $ conda env create -f environment.yml
   $ pip install .

Or simply install using ``pip`` exclusively::

   $ pip install .


.. toctree::
   :hidden:
   :caption: Modules

   modules/unlockgnn.model
   modules/unlockgnn.kernel_layers
   modules/unlockgnn.megnet_utils
   modules/unlockgnn.metrics
   modules/unlockgnn.initializers
