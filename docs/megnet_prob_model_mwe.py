"""Add uncertainty quantification to a MEGNetModel for predicting formation energies."""
from pathlib import Path

from megnet.models import MEGNetModel
from unlockgnn.download import load_data
from unlockgnn.model import MEGNetProbModel

MP_API_KEY: str = ""  # Set this to a Materials Project API key

TRAINING_RATIO: float = 0.8
NUM_INDUCING_POINTS: int = 500
BATCH_SIZE: int = 128
MODEL_SAVE_DIR: Path = Path("binary_e_form_model")

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
kl_weight = BATCH_SIZE / num_training
prob_model = MEGNetProbModel(
    num_inducing_points=NUM_INDUCING_POINTS,
    save_path=MODEL_SAVE_DIR,
    meg_model=meg_model,
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
    prob_model.save()


# 3. First training run is to approximate correct inducing points locations
train_model()
# 4. Unfreeze GNN layers and train again for fine tuning
prob_model.set_frozen("GNN", freeze=False)
train_model()
# 5. ``train_model`` also handles saving.
