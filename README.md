# unlockGNN

[![Documentation Status](https://readthedocs.org/projects/unlockgnn/badge/?version=latest)](https://unlockgnn.readthedocs.io/en/latest/?badge=latest)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Coverage Status](https://coveralls.io/repos/github/a-ws-m/unlockGNN/badge.svg?branch=master)](https://coveralls.io/github/a-ws-m/unlockGNN?branch=master)

A Python package for interpreting and extracting uncertainties in graph neural network models of chemical systems based upon Gaussian processes.

## Statement of need

Graph neural networks (GNNs) are powerful tools for performing materials property prediciton based on structural information.
They offer a cheaper alternative to DFT models and are therefore promising for high throughput screening of materials.
However, current implementations of GNNs lack uncertainty quantifiers for regression problems.
Knowledge of the certainty in an estimate is particularly important for data-driven predictive models,
as the reliability of a prediction depends on the existence of functionally similar structures in the
training dataset, which cannot be readily determined.

We have developed utilities for training a neural network-fed Gaussian process as an uncertainty quantifier.
The framework enables the training of a precursor GNN, which functions as a representation learning algorithm.
A layer of the GNN can then be selected to serve as the input (index points) for a Gaussian process.
The model can be saved and reloaded in a bundled format and used to perform predictions and confidence intervals
on unseen structures.

## Installation

The package can be installed by cloning this repository and building it using either anaconda or pip.
In the root directory of the package, run one of either

```pip install .``` or

```conda env install -f environment.yml```

In addition, `tensorflow` must be installed as a separate dependency.
This is to give the user the choice of installing GPU support.
TensorFlow may be installed using either `pip install tensorflow` or `pip install tensorflow-cpu`.

The `dev_environment.yml` contains additional dependencies for development, testing and building documentation.
It can be installed using `conda env install -f dev_environment.yml`.

## Documentation

Full documentation is available for the project [here](https://unlockgnn.readthedocs.io/en/latest/).

## License and attribution

Code licensed under the MIT License.

## Development notes

### Reporting issues

Please use the Issue tracker to report bugs in the software, suggest feature improvements, or seek support.

### Contributing to unlockGNN

Contributions are very welcome as we look to make unlockGNN more flexible and efficient.
Please use the [Fork and Pull](https://guides.github.com/activities/forking/) workflow to make contributions and follow the contribution guidelines:

- Use the environment defined in `dev_environment.yml`. This installs `black`, the formatter used for this project, as well as utilities for building documentation, enabling the testing suite and publishing to PyPi.
- Write tests for new features in the appropriate directory. Use `@pytest.mark.slow` for slow tests.
- Use [Google-style Docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html). Check docstrings with `pydocstyle`.
- Feel free to clean up others' code as you go along.

### List of developers

Contributors to unlockGNN:

- [Alexander Moriarty](https://github.com/a-ws-m)

Huge thanks to [Keith Butler](https://github.com/keeeto), [Aron Walsh](https://github.com/aronwalsh) and Kazuki Morita for supervising the project at its inception and for their immense support.
