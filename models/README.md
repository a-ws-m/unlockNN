# Example models

These models are included as a convenience for testing unlockGNN and demonstrating
its usage with more minimalist examples. They can be loaded using
`unlockgnn.download.load_model(fname)`, where `fname` is the name of the model
file _without the file extensions_.

Currently, the only included model is `binary_e_form`, which was trained to
predict formation energies per atom (in eV) on binary compounds that lie on the
convex hull and their formation energies in eV, from the
[Materials Project](https://materialsproject.org/).
The training dataset is also included in this repository under `data/binary_e_form.pkl`.
