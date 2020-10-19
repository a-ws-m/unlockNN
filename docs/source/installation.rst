Installation
============

The package can be installed from source using either anaconda or pip.
In the root directory of the package, run one of either
``pip install .`` or
``conda env create -f environment.yml``.

In addition, ``tensorflow`` must be installed as a separate dependency.
This is to give the user the choice of installing GPU support.
TensorFlow may be installed using either ``pip install tensorflow`` or ``pip install tensorflow-cpu``.

The ``dev_environment.yml`` contains additional dependencies for development, testing and building documentation.
It can be installed using ``conda env create -f dev_environment.yml``.
