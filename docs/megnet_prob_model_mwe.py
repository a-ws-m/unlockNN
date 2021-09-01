"""Add uncertainty quantification to a MEGNetModel for predicting formation energies."""
from pathlib import Path

from megnet.models import MEGNetModel
from unlockgnn.download import load_data
from unlockgnn.model import MEGNetProbModel

TRAINING_RATIO: float = 0.8
NUM_INDUCING_POINTS: int = 500  # Number of inducing index points for VGP
BATCH_SIZE: int = 128
MODEL_SAVE_DIR: Path = Path("binary_e_form_example")

# Data preprocessing:
# Load binary compounds' formation energies example data,
# then split into training and validation subsets.
full_df = load_data("binary_e_form")
num_training = int(TRAINING_RATIO * len(full_df.index))
train_df = full_df[:num_training]
val_df = full_df[num_training:]
# 4217 training samples, 1055 validation samples.

train_structs = train_df["structure"]
val_structs = val_df["structure"]
train_targets = train_df["formation_energy_per_atom"]
val_targets = val_df["formation_energy_per_atom"]

# 1. Load MEGNetModel
meg_model = MEGNetModel.from_mvl_models("Eform_MP_2019")

# 2. Make probabilistic model
# Specify Kullback-Leibler divergence weighting in loss function:
kl_weight = BATCH_SIZE / num_training
# Then make the model:
prob_model = MEGNetProbModel(
    meg_model=meg_model,
    num_inducing_points=NUM_INDUCING_POINTS,
    kl_weight=kl_weight,
)


def train_model():
    """Train and save the probabilistic model."""
    prob_model.train(
        train_structs,
        train_targets,
        epochs=50,
        val_structs=val_structs,
        val_targets=val_targets,
    )
    prob_model.save(MODEL_SAVE_DIR)


# 3. First training run is to approximate correct inducing points locations
train_model()
# 4. Unfreeze GNN layers and train again for fine tuning
prob_model.set_frozen("GNN", freeze=False)
train_model()
# 5. ``train_model`` also handles saving.

# We can then load the model from disk and perform some predictions
loaded_model = MEGNetProbModel.load(MODEL_SAVE_DIR)
example_struct, example_energy = train_structs[0], train_targets[0]
predicted, stddev = loaded_model.predict(example_struct)
# Two standard deviations is the 95% confidence interval
print(f"{example_struct.composition}: ")
print(f"Predicted E_f: {predicted.item():.3f} ± {stddev.item() * 2:.3f} eV")
print(f"Actual E_f: {example_energy:.3f} eV")
"""La2 Rh2: 
Predicted E_f: -0.739 ± 0.063 eV
Actual E_f: -0.737 eV"""
