# Example scripts

This folder contains example usage scripts for `UnlockGNN`.
With the exception of `joint_model.py`, the scripts follow an example workflow for creating a model for predicting
solid state energies of binary compounds' constituent species, from data collection and sanitisation, to creating
a precursor graph network model using `MEGNet`, to finally adding uncertainty quantification using `UnlockGNN`.
`joint_model.py` is a standalone script file that shows how to build a similar model for a benchmark dataset
in just a few lines using `UnlockGNN`'s `MEGNetProbModel` class: a wrapper that takes care of a lot of the
data handling that is needed to make such a model otherwise.

It is highly recommended to run these examples on a machine with a GPU and using `tensorflow-gpu`.
Some script files make use of [`MLFlow`](https://mlflow.org/) to provide tracking of model performance during training,
but this is optional.

## MEGNetProbModel (`joint_model.py`)

This file contains an example script for creating a `MEGNetProbModel`, an implementation of a `MEGNetModel`
that also provides uncertainty quantification for predictions.
The model is trained on [`matminer`](https://hackingmaterials.lbl.gov/matminer/)'s `matbench_perovskite` data set
in order to predict formation energies.

## Predicting solid state energies (SSEs)

### Data mining (`get_sse_db.py`)

This file demonstrates the process of downloading and extracting solid state energies using `UnlockGNN`'s `datalib` module.

### Training a precursor graph network (`train_base_megnet.py`)

Next, the code in this file trains a `MEGNetModel` to predict SSEs of the binary compounds as a vector of `(cation_sse, anion_sse)`.

### Training a graph network-fed Gaussian process (`train_gp.py` and `train_post_vgp.py`)

Finally, each of these files demonstrates how to train the uncertainty quantifier using the precursor model.
The key differences between each of the techniques are the cost function, the kernel function and the data type of each model:

- **Variational Gaussian Process (VGP)** makes use of the [`VariationalGaussianProcess`](https://www.tensorflow.org/probability/api_docs/python/tfp/layers/VariationalGaussianProcess)
Keras layer and uses a [variational loss function](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/VariationalGaussianProcess#variational_loss).
It uses a radial basis function kernel.
- **Gaussian Process (GP)** inherits from [`tf.Module`](https://www.tensorflow.org/api_docs/python/tf/Module)
and uses negative log likelihood loss and a [Matern kernel with parameter 1/2](https://www.tensorflow.org/api_docs/python/tf/Module).
