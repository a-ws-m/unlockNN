# Pre-trained models for testing

This folder contains `MEGNetProbModel`s at various stages at training, for unit testing purposes.
All of the models herein use `VGP`s due to numerical instabilities when using `GP`s on a dataset of this size.
The models were trained on the `matbench_phonons` [dataset](https://ml.materialsproject.org/projects/matbench_phonons/).
The indices correspond to the `training_stage` property:

0. No training
1. `MEGNetModel` trained.
2. `VGP` trained.
