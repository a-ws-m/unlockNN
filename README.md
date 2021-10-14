# unlockNN

[![Documentation Status](https://readthedocs.org/projects/unlockgnn/badge/?version=latest)](https://unlockgnn.readthedocs.io/en/latest/?badge=latest)
![PyPi](https://img.shields.io/pypi/v/unlockNN)
[![status](https://joss.theoj.org/papers/b00df538a159c4b6816ec24d4d1716fb/status.svg)](https://joss.theoj.org/papers/b00df538a159c4b6816ec24d4d1716fb)
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

UnlockNN contains utilities for adding uncertainty quantification to Keras-based
models. This is achieved by replacing the last layer of the model with a
*variational Gaussian process* (VGP), a modification of a Gaussian process that
improves scalability to larger data sets. The caveat is that the modified model
must undergo further training in order to calibrate the uncertainty quantifier;
however, this typically only requires a small number of training iterations.

UnlockNN also contains a specific configuration for adding uncertainty
quantification to [MEGNet](https://github.com/materialsvirtuallab/megnet/): a
powerful graph NN model for predicting properties of molecules and crystals.

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

- Use the environment defined in `dev_environment.yml`. This installs `black`, the formatter used for this project, as well as utilities for building documentation (`sphinx` and the `insegel` theme), enabling the testing suite (`pytest` and `pytest-cov`) and publishing to PyPi (`build`, but this will be handled by the package maintainer).
- Use `black` to format all Python files that you edit: `black {edited_file.py}` or `python -m black {edited_file.py}`.
- Write tests for new features in the appropriate directory. Run tests using `pytest tests/`, or optionally with `pytest --cov=unlocknn tests/` to generate coverage on the fly.
- After testing that `pytest` works for your current environment, run `tox` in the root directory of the project to check that all versions of Python are compatible.
- Use [Google-style Docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html). Check docstrings with `pydocstyle`: `pydocstyle {edited_file.py}`.
- Check docstrings are valid Sphinx RST and that the documentation compiles without errors: in the `docs` directory, run `make html`.
- Feel free to clean up others' code as you go along.

### List of developers

Contributors to unlockNN:

- [Alexander Moriarty](https://github.com/a-ws-m)

Huge thanks to [Keith Butler](https://github.com/keeeto), [Aron Walsh](https://github.com/aronwalsh) and [Kazuki Morita](https://github.com/KazMorita) for supervising the project at its inception and for their immense support.
