# unlockGNN

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

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

## License and attribution

Code licensed under the MIT License.

## Development notes

### List of developers

Contributors to unlockGNN:
    - [Alexander Moriarty](https://github.com/a-ws-m)

Huge thanks to [Keith Butler](https://github.com/keeeto), [Aron Walsh](https://github.com/aronwalsh) and Kazuki Morita for supervising the project at its inception and providing support.
