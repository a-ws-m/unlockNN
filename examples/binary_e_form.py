"""Binary compound formation energy prediction example.

This notebook demonstrates how to create a probabilistic model for predicting
formation energies of binary compounds with a quantified uncertainty. Before
running this notebook, ensure that you have a valid Materials Project API key
from <https://www.materialsproject.org/dashboard>. Next, either put this
key in a `.config` file, or change `MAPI_KEY` to the key.

Be careful not to include API keys in published versions of this notebook!

"""

import shutil
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from megnet.models import MEGNetModel
from pymatgen.ext.matproj import MPRester
from tensorflow.keras.callbacks import TensorBoard
from unlockgnn import MEGNetProbModel
from unlockgnn.initializers import SampleInitializer
from unlockgnn.metrics import evaluate_uq_metrics, AVAILABLE_METRICS

THIS_DIR = Path(".").parent
CONFIG_FILE = THIS_DIR / ".config"

MAPI_KEY = None
MODEL_SAVE_DIR: Path = THIS_DIR / "binary_e_form_model"
DATA_SAVE_DIR: Path = THIS_DIR / "binary_data.pkl"
LOG_DIR = THIS_DIR / "logs"
BATCH_SIZE: int = 128
NUM_INDUCING_POINTS: int = 2000

parser = ArgumentParser()
parser.add_argument("--train", action="store_true", dest="train")
parser.add_argument("--eval", action="store_true", dest="evaluate")
parser.add_argument("--overwrite", action="store_true", dest="overwrite")
args = parser.parse_args()

train: bool = args.train
evaluate: bool = args.evaluate
overwrite: bool = args.overwrite

try:
    mp_key = CONFIG_FILE.read_text()
except FileNotFoundError:
    if MAPI_KEY is None:
        raise ValueError(
            "Enter Materials Project API key either in a `.config` file or in the notebook itself."
        )
    mp_key = MAPI_KEY

# # Data gathering
#
# Here we download binary compounds that lie on the convex hull from the Materials
# Project, then split them into training and validation subsets.
#

query = {
    "criteria": {"nelements": 2, "e_above_hull": 0},
    "properties": ["structure", "formation_energy_per_atom"],
}

if DATA_SAVE_DIR.exists():
    full_df = pd.read_pickle(DATA_SAVE_DIR)
else:
    with MPRester(mp_key) as mpr:
        full_df = pd.DataFrame(mpr.query(**query))
    full_df.to_pickle(DATA_SAVE_DIR)


print(full_df.head())


TRAINING_RATIO: float = 0.8

num_training = int(TRAINING_RATIO * len(full_df.index))
train_df = full_df[:num_training]
val_df = full_df[num_training:]

print(f"{num_training} training samples, {len(val_df.index)} validation samples.")

# # Model creation
#
# Now we load the `MEGNet` 2019 formation energies model, then convert this to a
# probabilistic model. We begin by first training this `MEGNetModel` on our data to
# achieve a slightly more precise fit.
#

meg_model = MEGNetModel.from_mvl_models("Eform_MP_2019")


train_structs = train_df["structure"]
val_structs = val_df["structure"]

train_targets = train_df["formation_energy_per_atom"]
val_targets = val_df["formation_energy_per_atom"]


# Make the initializer
# index_points_init = SampleInitializer(train_structs, meg_model)
index_points_init = None


KL_WEIGHT = BATCH_SIZE / num_training

if overwrite:
    for directory in [MODEL_SAVE_DIR, LOG_DIR]:
        shutil.rmtree(directory)
    prob_model = MEGNetProbModel(
        num_inducing_points=NUM_INDUCING_POINTS,
        save_path=MODEL_SAVE_DIR,
        meg_model=meg_model,
        kl_weight=KL_WEIGHT,
        index_initializer=index_points_init,
    )
else:
    prob_model = MEGNetProbModel.load(MODEL_SAVE_DIR)

# # Train the uncertainty quantifier
#
# Now we train the model. By default, the `MEGNet` (GNN) layers of the model are
# frozen after initialization. Therefore, when we call `prob_model.train()`, the
# only layers that are optimized are the `VariationalGaussianProcess` (VGP) and the
# `BatchNormalization` layer (`Norm`) that feeds into it.
#
# After this initial training, we will then fine tune the model by freezing the
# `Norm` and VGP layers and training just the GNN layers. Then, finally, we
# unfreeze _all_ the layers and train the full model simulateously.
#

if train:
    tb_callback_1 = TensorBoard(log_dir=LOG_DIR / "vgp_training", write_graph=False)
    tb_callback_2 = TensorBoard(log_dir=LOG_DIR / "fine_tuning", write_graph=False)

    print("Training VGP...")
    prob_model.train(
        train_structs,
        train_targets,
        epochs=50,
        val_structs=val_structs,
        val_targets=val_targets,
        callbacks=[tb_callback_1],
    )
    prob_model.save()

    prob_model.set_frozen(["GNN", "VGP"], freeze=False)

    print("Fine tuning...")
    prob_model.train(
        train_structs,
        train_targets,
        epochs=50,
        val_structs=val_structs,
        val_targets=val_targets,
        callbacks=[tb_callback_2],
    )

    prob_model.save()

# # Model evaluation
#
# Finally, we'll evaluate model metrics and make some sample predictions! Note that the predictions give predicted values and standard deviations. The standard deviations can then be converted to an uncertainty;
# in this example, we'll take the uncertainty as twice the standard deviation, which will give us the 95% confidence interval (see <https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule>).
#

if evaluate:
    prob_model.evaluate(val_structs, val_targets)

    example_structs = val_structs[:10].tolist()
    example_targets = val_targets[:10].tolist()

    predicted, stddevs = prob_model.predict(example_structs)
    uncerts = 2 * stddevs

    pred_df = pd.DataFrame(
        {
            "Composition": [
                struct.composition.reduced_formula for struct in example_structs
            ],
            "Formation energy per atom / eV": example_targets,
            "Predicted / eV": [
                f"{pred:.2f} ± {uncert:.2f}" for pred, uncert in zip(predicted, uncerts)
            ],
        }
    )

    train_metrics = evaluate_uq_metrics(
        prob_model, train_structs, train_targets, AVAILABLE_METRICS.keys()
    )
    val_metrics = evaluate_uq_metrics(
        prob_model, val_structs, val_targets, AVAILABLE_METRICS.keys()
    )

    print(f"{train_metrics=}")
    print(f"{val_metrics=}")
