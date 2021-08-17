"""Example script for training a model to predict optical phonon modes."""
import gzip
import json
import logging
import os
import sys
from argparse import ArgumentParser
from math import floor
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import requests
from megnet.models import MEGNetModel
from pymatgen.core.structure import Structure
from unlockgnn import MEGNetProbModel
from tensorflow.keras.callbacks import TensorBoard
from unlockgnn.megnet_utils import default_megnet_config
from unlockgnn.metrics import MAE, NLL, evaluate_uq_metrics

HERE = Path(__file__).parent
PHONONS_URL = "https://ml.materialsproject.org/projects/matbench_phonons.json.gz"

TRAINING_RATIO: float = 0.8
PHONONS_SAVE_DIR = HERE / "phonons.pkl"
MEGNET_LOGS = HERE / "megnet_logs"
PROB_GNN_LOGS = HERE / "prob_logs"
MEGNET_MODEL_DIR = HERE / "meg_model"
PROB_MODEL_DIR = HERE / "prob_model"
METRICS_LOGS = HERE / "metrics.log"

for log_dir in [MEGNET_LOGS, PROB_GNN_LOGS]:
    if not log_dir.exists():
        os.mkdir(log_dir)

logging.basicConfig(
    filename=METRICS_LOGS, level=logging.INFO, style="%(asctime)s %(message)s"
)


def download_data(url: str, save_dir: Path) -> pd.Dataframe:
    """Download and extract data from the URL.

    Expects a `matbench` dataset URL.

    """
    if save_dir.exists():
        print(f"Loading data from {save_dir}...", flush=True)
        return pd.read_pickle(save_dir)

    print("Downloading and extracting data... ", flush=True, end=None)
    r = requests.get(url)
    json_data = gzip.decompress(r.content)
    dict_data = json.loads(json_data)
    print("Done!", flush=True)

    index = dict_data["index"]
    cols = dict_data["columns"]
    table_content = dict_data["data"]

    # Table content is a list of entries. Each entry is a list with two
    # elements: a pymatgen.Structure as a dict and our target value.
    pd_data: List[Tuple[Structure, float]] = []
    for entry in table_content:
        struct = Structure.from_dict(entry[0])
        target = entry[1]
        pd_data.append((struct, target))

    df = pd.DataFrame(pd_data, index, cols)
    print(f"Saving data to {save_dir}... ", end=None, flush=True)
    df.to_pickle(save_dir)
    print("Done!", flush=True)
    return df


def log_metrics(metrics: Dict[str, float], data_name: str):
    """Log all metrics from a dictionary."""
    for metric_name, value in metrics.items():
        logging.info("ProbGNN %s %s = %f", data_name, metric_name, value)


def main() -> None:
    """Execute main script."""
    parser = ArgumentParser()
    parser.add_argument(
        "--train",
        action="store_true",
        help="Whether to train the model.",
        dest="do_train",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Whether to evaluate the model.",
        dest="do_eval",
    )
    parser.add_argument(
        "--which",
        nargs="+",
        choices=["MEGNet", "VGP", "ProbGNN"],
        required=("--train" in sys.argv),
        help=(
            "Which components to train: "
            "MEGNet -- Just the MEGNetModel; "
            "VGP -- Just the VGP part of the ProbGNN; "
            "ProbGNN -- The whole ProbGNN."
        ),
        dest="which",
    )
    parser.add_argument(
        "--epochs",
        "-n",
        type=int,
        required=("--train" in sys.argv),
        help="Number of training epochs.",
        dest="epochs",
    )
    parser.add_argument(
        "--inducing",
        "-i",
        type=int,
        help="Number of inducing index points.",
        default=500,
        dest="num_inducing",
    )
    args = parser.parse_args()

    do_train: bool = args.do_train
    do_eval: bool = args.do_eval
    which_model: str = args.which
    epochs: int = args.epochs
    num_inducing: int = args.num_inducing

    # Load the MEGNetModel into memory
    try:
        meg_model: MEGNetModel = MEGNetModel.from_file(str(MEGNET_MODEL_DIR))
    except FileNotFoundError:
        meg_model = MEGNetModel(**default_megnet_config())

    # Load the data into memory
    df = download_data(PHONONS_URL, PHONONS_SAVE_DIR)
    structures = df["structure"]
    targets = df["last phdos peak"]
    num_data = len(structures)
    print(f"{num_data} datapoints loaded.")

    num_training = floor(num_data * TRAINING_RATIO)
    print(f"{num_training} training data, {num_data-num_training} test data.")
    train_structs = structures[:num_training]
    train_targets = targets[:num_training]
    test_structs = structures[num_training:]
    test_targets = targets[num_training:]

    if which_model == "MEGNet":
        if do_train:
            tf_callback = TensorBoard(MEGNET_LOGS, write_graph=False)
            meg_model.train(
                train_structs,
                train_targets,
                test_structs,
                test_targets,
                automatic_correction=False,
                dirname="meg_checkpoints",
                epochs=epochs,
                callbacks=[tf_callback],
            )
        if do_eval:
            train_predicted = meg_model.predict_structures(train_structs)
            train_mae = MAE(train_predicted, None, train_targets)
            logging.info("MEGNet train MAE = %f", train_mae)

            test_predicted = meg_model.predict_structures(test_structs)
            test_mae = MAE(test_predicted, None, test_targets)
            logging.info("MEGNet test MAE = %f", test_mae)
    else:
        # Load the ProbGNN into memory
        try:
            prob_model: MEGNetProbModel = MEGNetProbModel.load(PROB_MODEL_DIR)
        except FileNotFoundError:
            prob_model = MEGNetProbModel(
                num_inducing, PROB_MODEL_DIR, meg_model, metrics=["MAE", NLL()]
            )

        if do_train:
            if which_model == "VGP":
                prob_model.set_frozen("GNN", recompile=False)
                prob_model.set_frozen(["VGP", "Norm"], freeze=False)
                tf_callback = TensorBoard(PROB_GNN_LOGS / "vgp_only", write_graph=False)
            else:
                prob_model.set_frozen("GNN", freeze=False, recompile=False)
                prob_model.set_frozen(["VGP", "Norm"])
                tf_callback = TensorBoard(PROB_GNN_LOGS / "probgnn", write_graph=False)
            prob_model.train(
                train_structs,
                train_targets,
                epochs,
                test_structs,
                test_targets,
                callbacks=[tf_callback],
            )
        if do_eval:
            train_metrics = evaluate_uq_metrics(
                prob_model, train_structs, train_targets
            )
            log_metrics(train_metrics, "training")
            test_metrics = evaluate_uq_metrics(prob_model, test_structs, test_targets)
            log_metrics(test_metrics, "test")


if __name__ == "__main__":
    main()
