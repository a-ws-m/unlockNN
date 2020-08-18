.. Neural UQ documentation master file, created by
   sphinx-quickstart on Mon Aug 17 15:09:19 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Neural UQ
=========

Neural UQ is a package for adding uncertainty quantification to Keras models using
Gaussian processes.
It is a Python library that provides tools for extracting layer outputs from a
pre-trained Keras model and using them as index points for a Gaussian process,
which can be trained to predict a distribution of likely target values,
rather than a single explicit prediction.
The distribution's mean and standard deviation can be used as a prediction and
confidence interval for predictive models.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   installation
   quickstart
   usage
   modules



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
