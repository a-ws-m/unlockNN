# unlockNN

[![Documentation Status](https://readthedocs.org/projects/unlocknn/badge/?version=latest)](https://unlockgnn.readthedocs.io/en/latest/?badge=latest)
![PyPi](https://img.shields.io/pypi/v/unlockNN)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![codecov](https://codecov.io/gh/a-ws-m/unlockNN/branch/master/graph/badge.svg?token=TBDX3P6OZ3)](https://codecov.io/gh/a-ws-m/unlockNN)

A Python package for interpreting and extracting uncertainties in neural network models of chemical systems based upon Gaussian processes.

## Statement of need

Neural networks (NNs) are powerful tools for materials property prediciton (MPP)
based on structural information. After training, they offer a cheaper
alternative to density function theory (DFT) and are therefore promising for
high throughput screening of materials. However, most current implementations of
NNs for MPP lack uncertainty quantifiers. Knowledge of the certainty in an
estimate is particularly important for machine learning models, as the
reliability of a prediction depends on the existence of functionally similar
structures in the training dataset, which cannot be readily determined.

UnlockNN contains utilities for training a NN-fed Gaussian process as an
uncertainty quantifier. The framework enables the training of a precursor NN,
which functions as a representation learning algorithm. A layer of the NN can
then be selected to serve as the input (index points) for a Gaussian process.
The model can be saved and reloaded in a bundled format and used to perform
predictions and confidence intervals on unseen structures.

## Installation


The package can be installed by cloning this repository and building it using either anaconda or pip,
or it can be downloaded directly from PyPi.

To install from PyPi, run `pip install unlockNN`.
To install from source:

```bash
git clone https://github.com/a-ws-m/unlockNN.git
cd unlockNN
conda env create -f environment.yml  # Optional: create a virtual environment with conda
pip install .
```

The `dev_environment.yml` contains additional dependencies for development, testing and building documentation.
It can be installed using `conda env create -f dev_environment.yml`.

## Documentation

Full documentation is available for the project [here](https://unlockgnn.readthedocs.io/en/latest/).

## License and attribution

Code licensed under the MIT License.

## Development notes

### Reporting issues

Please use the Issue tracker to report bugs in the software, suggest feature improvements, or seek support.

### Contributing to unlockNN

Contributions are very welcome as we look to make unlockNN more flexible and efficient.
Please use the [Fork and Pull](https://guides.github.com/activities/forking/) workflow to make contributions and follow the contribution guidelines:

- Use the environment defined in `dev_environment.yml`. This installs `black`, the formatter used for this project, as well as utilities for building documentation, enabling the testing suite and publishing to PyPi.
- Write tests for new features in the appropriate directory.
- Use [Google-style Docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html). Check docstrings with `pydocstyle`.
- Feel free to clean up others' code as you go along.

### List of developers

Contributors to unlockNN:

- [Alexander Moriarty](https://github.com/a-ws-m)

Huge thanks to [Keith Butler](https://github.com/keeeto), [Aron Walsh](https://github.com/aronwalsh) and [Kazuki Morita](https://github.com/KazMorita) for supervising the project at its inception and for their immense support.
