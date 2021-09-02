.. unlockNN documentation master file, created by
   sphinx-quickstart on Wed Aug 18 15:33:30 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
.. module:: unlocknn

What is unlockNN?
==================

UnlockNN is a Python package for adding uncertainty quantification to
Keras models for materials property prediction.

This documentation was built on |today|.

Installation
------------

Using ``pip``::

   $ pip install unlockNN

From source
^^^^^^^^^^^

First, clone the repository::

   $ git clone https://github.com/a-ws-m/unlockNN.git
   $ cd unlockNN

Then, either install dependencies using ``conda``::

   $ conda env create -f environment.yml
   $ pip install .

Or simply install using ``pip`` exclusively::

   $ pip install .

.. toctree::
   :hidden:

   Home <self>
   quickstart
   GitHub <https://github.com/a-ws-m/unlockNN/>

.. toctree::
   :hidden:
   :caption: Modules

   modules/unlocknn.model
   modules/unlocknn.kernel_layers
   modules/unlocknn.metrics
   modules/unlocknn.megnet_utils
   modules/unlocknn.download
   modules/unlocknn.initializers
