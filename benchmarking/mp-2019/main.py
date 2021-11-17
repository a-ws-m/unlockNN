"""Train and evaluate a MEGNetProbModel for formation energy prediction on the MP-2019 dataset.

<https://figshare.com/articles/dataset/Graphs_of_Materials_Project_20190401/8097992>

"""
from datetime import datetime
import json
from argparse import ArgumentError, ArgumentParser
from os import mkdir
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

import numpy as np
import pandas as pd
from megnet.models import MEGNetModel
from megnet.utils.models import load_model
from pymatgen.core.structure import Structure
from tensorflow.python.keras.callbacks import TensorBoard
from unlocknn import MEGNetProbModel
from unlocknn.megnet_utils import MEGNetGraph
from unlocknn.metrics import evaluate_uq_metrics, MSE

HERE = Path(__file__).parent
DATA_FILE = HERE / "mp.2019.04.01.json"
MODELS_DIR = HERE / "models"
LOGS_DIR = HERE / "logs"

MEGNET_METRICS_LOG = HERE / "meg_metrics.log"
PROB_MODEL_METRICS_LOG = HERE / "prob_metrics.json"

TRAIN_RATIO: float = 0.9
VAL_RATIO: float = 0.05

for directory in [MODELS_DIR, LOGS_DIR]:
    if not directory.exists():
        mkdir(directory)

def model_dir(num_inducing_points: int) -> Path:
    """Get the models directory for the model with _n_ index points."""
    return MODELS_DIR / f"mp2019_e_form_{num_inducing_points}"

def log_dir(num_inducing_points: int) -> Path:
    """Get the logs directory for the model with _n_ index points."""
    log_dir = LOGS_DIR / f"mp2019_e_form_{num_inducing_points}"
    if not log_dir.exists():
        mkdir(log_dir)
    return log_dir

def get_tb_callback(num_inducing_points: int) -> TensorBoard:
    """Get a configured TensorBoard callback."""
    tensorboard_run_loc = log_dir(num_inducing_points) / str(
        datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    )

    # These parameters avoid segmentation fault
    return TensorBoard(
        tensorboard_run_loc, write_graph=False, profile_batch=0
    )

def load_meg_model() -> MEGNetModel:
    """Load the MEGNetModel."""
    return load_model("Eform_MP_2019")

def eval_meg_model(meg_model: MEGNetModel, graphs: Iterable[MEGNetGraph], targets: Iterable[float]):
    """Evaluate the MSE of a given MEGNetModel."""
    predictions = meg_model.predict_graphs(graphs)
    return MSE(predictions, None, targets)

def load_data() -> pd.DataFrame:
    """Load the data from disk."""
    with DATA_FILE.open("r") as f:
        data = json.load(f)
    
    df = pd.DataFrame({
        "graph": [entry["graph"] for entry in data],
        "e_form_per_atom": [entry["formation_energy_per_atom"] for entry in data],
    })
    # Decorate DataFrame with set type
    num_entries = len(df.index)
    train_size = np.floor(TRAIN_RATIO * num_entries)
    val_size = np.floor(VAL_RATIO * num_entries)

    df["training_set"] = [i < train_size for i in range(num_entries)]
    df["testing_set"] = [i >= train_size + val_size for i in range(num_entries)]
    df["validation_set"] = [not (train or test) for train, test in zip(df["training_set"], df["testing_set"])]

    # Sanity check
    test_size = num_entries - train_size - val_size
    assert df["training_set"].sum() == train_size
    assert df["validation_set"].sum() == val_size
    assert df["testing_set"].sum() == test_size
    for _, entry in df.iterrows():
        assert entry["training_set"] + entry["validation_set"] + entry["testing_set"] == 1

    print(f"{train_size=}")
    print(f"{val_size=}")
    print(f"{test_size=}")

    return df

def parse_args() -> Dict[str, Any]:
    """Parse CLI arguments."""
    parser = ArgumentParser()
    parser.add_argument("--meg", "-m", action="store_true", dest="meg", help="Evaluate the base MEGNetModel's performance.")
    parser.add_argument("--train", "-t", type=int, dest="train", help="Number of training iterations.")
    parser.add_argument("--eval", "-e", action="store_true", dest="eval", help="Set this flag to evaluate the model.")
    points_arg = parser.add_argument("--points", "-p", type=int, dest="points", help="The number of model inducing index points.")
    parser.add_argument("--comp", "-c", choices=["NN", "VGP"], nargs="*", default=["VGP"], dest="component", help="The component of the model to train.")

    args = parser.parse_args()
    meg: bool = args.meg
    train: Optional[int] = args.train
    evaluate: bool = args.eval
    points: Optional[int] = args.points
    comp: List[str] = args.component

    if train or evaluate:
        # Points must be specified
        if not points:
            raise ArgumentError(points_arg, "Must specify number of inducing points for model for training or evaluation.")

    return {"train": train, "eval": evaluate, "meg": meg, "points": points, "comp": comp}

def find_duplicate_weights(prob_model: MEGNetProbModel) -> Set[str]:
    """Find any duplicate weight names in a model."""
    names = [weight.name for layer in prob_model.model.layers for weight in layer.weights]
    dupe = set()
    seen = set()

    for name in names:
        if name in seen:
            dupe.add(name)
        seen.add(name)

    return dupe

def main():
    """Evaluate CLI arguments and execute main program flow."""
    cli_args = parse_args()

    print("Loading data...")
    df = load_data()
    train_df = df.query("training_set")
    val_df = df.query("validation_set")
    test_df = df.query("testing_set")

    print("Loaded data:")
    print(train_df.describe())
    print(train_df.head())
    print(val_df.describe())
    print(val_df.head())
    print(test_df.describe())
    print(test_df.head())

    if cli_args["points"]:
        NUM_INDUCING_POINTS: int = cli_args["points"]
        MODEL_DIR = model_dir(NUM_INDUCING_POINTS)

    if cli_args["meg"]:
        # * Evaluate MEGNet
        print("Evaluating MEGNet...")
        meg_model = load_meg_model()
        test_mse = eval_meg_model(meg_model, test_df["graph"], test_df["e_form_per_atom"])
        MEGNET_METRICS_LOG.write_text(f"{test_mse=}")
        print(f"Wrote evaluation results to {MEGNET_METRICS_LOG}")

    if cli_args["train"] or cli_args["eval"]:
        try:
            prob_model = MEGNetProbModel.load(MODEL_DIR)

            dupes = find_duplicate_weights(prob_model)
            print(f"Duplicate names after loading: {dupes}")

        except:
            if cli_args["eval"]:
                raise ValueError("Couldn't load model; nothing to evaluate")
            else:
                meg_model = load_meg_model()
                prob_model = MEGNetProbModel(meg_model, 100)
                prob_model.save(MODEL_DIR, ckpt_path=None)

    if cli_args["train"]:
        # * Freeze layers
        freezable = ["VGP", "NN"]
        to_freeze = [layer for layer in freezable if layer not in cli_args["comp"]]

        print(f"{to_freeze=}")
        print(f"{cli_args['comp']=}")
        print(f"{type(cli_args['comp'])=}")

        dupes = find_duplicate_weights(prob_model)
        print(f"Duplicate names after computing which to freeze: {dupes}")

        prob_model.set_frozen(to_freeze, recompile=False)

        dupes = find_duplicate_weights(prob_model)
        print(f"Duplicate names after freezing to_freeze: {dupes}")

        prob_model.set_frozen(cli_args["comp"], freeze=False)

        dupes = find_duplicate_weights(prob_model)
        print(f"Duplicate names after thawing : {dupes}")
        raise ValueError("Duplicate weight names found.")

        # * Train the probabilistic model
        prob_model.train(train_df["graph"].values, train_df["e_form_per_atom"], cli_args["train"], val_df["graph"].values, val_df["e_form_per_atom"], callbacks=[get_tb_callback(NUM_INDUCING_POINTS)])
        prob_model.save(MODEL_DIR)
    
    if cli_args["eval"]:
        # * Evaluate prob_model
        train_df = df.query("training_set")
        test_metrics = evaluate_uq_metrics(prob_model, test_df["graph"].values, test_df["e_form_per_atom"])
        with PROB_MODEL_METRICS_LOG.open("w") as f:
            json.dump(test_metrics, f)

if __name__ == "__main__":
    main()
