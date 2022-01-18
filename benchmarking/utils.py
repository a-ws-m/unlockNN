"""Train and evaluate a MEGNetProbModel for formation energy prediction on the MP-2019 dataset.

<https://figshare.com/articles/dataset/Graphs_of_Materials_Project_20190401/8097992>

"""
from abc import ABC, abstractmethod
from argparse import ArgumentError, ArgumentParser
from datetime import datetime
from os import mkdir
from pathlib import Path
from typing import Iterable, List, NamedTuple, Optional, Tuple

import pandas as pd
from megnet.models import MEGNetModel
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
from unlocknn import MEGNetProbModel
from unlocknn.megnet_utils import MEGNetGraph, Targets, default_megnet_config
from unlocknn.metrics import MAE, MSE, evaluate_uq_metrics


def eval_meg_model(
    meg_model: MEGNetModel, graphs: Iterable[MEGNetGraph], targets: Iterable[float]
) -> Tuple[float, float]:
    """Evaluate the MSE and MAE of a given MEGNetModel."""
    predictions = meg_model.predict_graphs(graphs)
    return MSE(predictions, None, targets), MAE(predictions, None, targets)


def train_megnet(
    model: MEGNetModel,
    num_epochs: int,
    train_input,
    train_targets,
    val_input,
    val_targets,
    checkpoint_path: Path,
    verbose: int = 2,
    callbacks=[],
    batch_size: int = 32,
):
    """Train a MEGNetModel."""
    ckpt_callback = ModelCheckpoint(
        checkpoint_path, save_best_only=True, save_weights_only=True
    )
    callbacks.append(ckpt_callback)
    model.train_from_graphs(
        train_graphs=train_input,
        train_targets=train_targets,
        validation_graphs=val_input,
        validation_targets=val_targets,
        epochs=num_epochs,
        save_checkpoint=False,
        automatic_correction=False,
        callbacks=callbacks,
        verbose=verbose,
        batch_size=batch_size,
    )


class Dataset(NamedTuple):
    """Container for datasets split into train, validation and test subsets."""

    train_input: List[MEGNetGraph]
    train_targets: Targets
    val_input: List[MEGNetGraph]
    val_targets: Targets
    test_input: List[MEGNetGraph]
    test_targets: Targets


class UnlockTrainer(ABC):
    """Handler for training and benchmarking unlockNN models."""

    def __init__(self, batch_size: int = 32) -> None:
        """Initialize parameters."""
        super().__init__()

        self.batch_size = batch_size

        # * Read command line arguments
        self.init_cli_args()

        # * Create necessary folders
        self.create_dir_structure()

        # * Initialize datasets, must be populated with `self.load_data`
        self.data: Optional[Dataset] = None

        # * Initialize model variables
        self.meg_model: Optional[MEGNetModel] = None
        self.prob_model: Optional[MEGNetProbModel] = None

    def create_dir_structure(self):
        """Create the project directory tree."""
        root_models_dir = self.root_dir / "models"
        self.megnet_models_dir = root_models_dir / "megnet"
        self.prob_models_dir = root_models_dir / "prob"

        root_logs_dir = self.root_dir / "logs"
        self.megnet_logs_dir = root_logs_dir / "megnet"
        self.prob_logs_dir = root_logs_dir / "prob"

        for dirname in [
            root_models_dir,
            self.megnet_models_dir,
            self.prob_models_dir,
            root_logs_dir,
            self.megnet_logs_dir,
            self.prob_logs_dir,
        ]:
            if not dirname.exists():
                mkdir(dirname)

    def execute(self):
        """Execute main program flow."""
        # * Ensure we're not training after we've evaluated test metrics
        if self.test_result_path.exists() and self.train:
            raise ArgumentError(
                None,
                f"The model has already been evaluated on the test set ({self.test_result_path})"
                "and cannot be trained any more.",
            )

        # * Start by loading data
        self.data = self.load_data(download_only=self.data_only)
        if self.data_only:
            return

        if self.meg:
            # * Handle MEGNetModel creation, training and evaluation
            if not self.load_meg_model():
                if self.evaluate:
                    raise ArgumentError(
                        None, f"No MEGNetModel found at {self.model_dir}"
                    )
                # Create model
                self.meg_model = MEGNetModel(**default_megnet_config(), loss="mae")

            if self.train:
                # * Training routine
                train_megnet(
                    self.meg_model,
                    self.train,
                    self.data.train_input,
                    self.data.train_targets,
                    self.data.val_input,
                    self.data.val_targets,
                    self.checkpoint_dir,
                    self.verbosity,
                    [self.tb_callback],
                    self.batch_size,
                )
                self.meg_model.save_model(str(self.model_dir))

            if self.val:
                # * Validation
                val_mse, val_mae = eval_meg_model(
                    self.meg_model, self.data.val_input, self.data.val_targets
                )
                meg_val_metrics = pd.Series({"mse": val_mse, "mae": val_mae})
                print(meg_val_metrics)

            if self.evaluate:
                # * Evaluation routine
                mse, mae = eval_meg_model(
                    self.meg_model, self.data.test_input, self.data.test_targets
                )
                meg_metrics = pd.Series({"mse": mse, "mae": mae})
                meg_metrics.to_csv(self.test_result_path)
                print(meg_metrics)

        else:
            # * Handle MEGNetProbModel creation, training and evaluation
            if self.model_dir.exists():
                self.prob_model = MEGNetProbModel.load(
                    self.model_dir, load_ckpt=(not self.ignore_ckpt)
                )
            else:
                if self.evaluate:
                    raise ArgumentError(
                        None, f"No MEGNetProbModel found at {self.model_dir}"
                    )
                # Create model
                if not self.load_meg_model():
                    raise ArgumentError(
                        None,
                        f"Cannot make new MEGNetProbModel: no MEGNetModel at {self.meg_model_dir}",
                    )
                self.prob_model = MEGNetProbModel(self.meg_model, self.points)

            if self.train:
                self.handle_freezing()
                self.prob_model.train(
                    self.data.train_input,
                    self.data.train_targets,
                    self.train,
                    self.data.val_input,
                    self.data.val_targets,
                    callbacks=[self.tb_callback],
                    ckpt_path=self.checkpoint_dir,
                    batch_size=self.batch_size,
                    verbose=self.verbosity,
                )
                self.prob_model.save(self.model_dir, self.checkpoint_dir)

            if self.val:
                prob_val_metrics = pd.Series(
                    evaluate_uq_metrics(
                        self.prob_model, self.data.val_input, self.data.val_targets
                    )
                )
                prob_val_metrics.to_csv(self.val_prob_metrics_dir)
                print(prob_val_metrics)

            if self.evaluate:
                prob_metrics = pd.Series(
                    evaluate_uq_metrics(
                        self.prob_model, self.data.test_input, self.data.test_targets
                    )
                )
                prob_metrics.to_csv(self.test_result_path)
                print(prob_metrics)

    def handle_freezing(self):
        """Freeze the appropriate probabilistic layers."""
        freezable = ["VGP", "NN"]
        to_freeze = [layer for layer in freezable if layer not in self.comp]
        self.prob_model.set_frozen(self.comp, freeze=False, recompile=False)
        self.prob_model.set_frozen(to_freeze, recompile=True)

    def load_meg_model(self) -> Optional[MEGNetModel]:
        """Load a MEGNetModel from disk."""
        # Check if model exists
        if self.meg_model_dir.exists():
            # Load it
            self.meg_model = MEGNetModel.from_file(str(self.meg_model_dir))
        if self.checkpoint_dir.exists() and not self.ignore_ckpt:
            self.meg_model.model.load_weights(self.checkpoint_dir)
            print(f"Loaded weights from {self.checkpoint_dir}")
        return self.meg_model

    @property
    @abstractmethod
    def task_name(self) -> str:
        """Human readable name segment to describe the name of this task."""
        ...

    @property
    def num_folds(self) -> Optional[int]:
        """Get the number of folds used for K-fold cross validation.

        Returns:
            num_folds: The number of folds, or ``None`` if the model doesn't
                use K-fold cross validation.

        """
        return None

    def init_cli_args(self):
        """Initialize properties from command line."""
        args = self.setup_argparse().parse_args()
        self.meg: bool = args.meg
        self.train: Optional[int] = args.train
        self.evaluate: bool = args.eval
        self.val: bool = args.val
        self.points: Optional[int] = args.points
        self.comp: List[str] = args.component
        self.verbosity: int = args.verbosity
        self.ignore_ckpt: bool = args.ignore_ckpt
        self.root_dir: Path = Path(args.root_dir)
        self.data_only: bool = args.data_only
        self.fold: int = args.fold
        if self.fold is None and not self.data_only:
            raise ValueError("Must supply `--fold` for training or evaluation.")

    def setup_argparse(self) -> ArgumentParser:
        """Set up expected command line arguments in order to decide what procedures to run."""
        parser = ArgumentParser(
            f"Script for model training and evaluation on {self.task_name}."
        )
        parser.add_argument(
            "--meg",
            "-m",
            action="store_true",
            dest="meg",
            help="Set this flag to work with just the base MEGNetModel, rather than the probabilistic model.",
        )
        parser.add_argument(
            "--train",
            "-t",
            type=int,
            dest="train",
            help="Number of training iterations.",
        )
        parser.add_argument(
            "--eval",
            "-e",
            action="store_true",
            dest="eval",
            help="Set this flag to evaluate the model.",
        )
        parser.add_argument(
            "--val",
            action="store_true",
            dest="val",
            help="Set this flag to evaluate the model on the validation data.",
        )
        parser.add_argument(
            "--points",
            "-p",
            type=int,
            dest="points",
            help="The number of model inducing index points.",
        )
        parser.add_argument(
            "--comp",
            "-c",
            choices=["NN", "VGP"],
            nargs="*",
            default=["VGP"],
            dest="component",
            help="The component of the model to train.",
        )
        parser.add_argument(
            "--verbosity",
            "-v",
            type=int,
            choices=range(3),
            default=1,
            dest="verbosity",
            help="The level of verbosity for Keras operations.",
        )
        parser.add_argument(
            "--ignore-ckpt",
            action="store_false",
            dest="ignore_ckpt",
            help=(
                "Whether to ignore saved checkpoints (corresponding to the best validation performance),"
                " preferring to load the latest saved weights."
            ),
        )
        parser.add_argument(
            "--root-dir",
            "-d",
            default=".",
            dest="root_dir",
            help="The directory to save models, logs and data.",
        )
        parser.add_argument(
            "--data-only",
            action="store_true",
            dest="data_only",
            help="If this flag is set, program will exit immediately after (down)loading data.",
        )

        if self.num_folds:
            parser.add_argument(
                "--fold",
                "-f",
                type=int,
                choices=range(self.num_folds),
                dest="fold",
                help="Which fold of data to use for training or evaluation.",
            )

        return parser

    @abstractmethod
    def load_data(self, download_only: bool = False) -> Optional[Dataset]:
        """Load data from disk for training/evaluation."""
        ...

    @property
    def model_dirname(self) -> str:
        """Get the name of the model's directory."""
        dirname = self.task_name
        if not self.meg:
            dirname += f"-{self.points}"
        if self.fold is not None:
            dirname += f"-{self.fold}"
        return dirname

    @property
    def meg_model_dir(self) -> Path:
        """Get the directory of the MEGNetModel specified by command line args."""
        return self.megnet_models_dir / self.model_dirname

    @property
    def model_dir(self) -> Path:
        """Get the models directory for the model specified by command line args."""
        return (
            self.megnet_models_dir if self.meg else self.prob_models_dir
        ) / self.model_dirname

    @property
    def log_dir(self) -> Path:
        """Get the logs directory for the model model specified by command line args."""
        return (
            self.megnet_logs_dir if self.meg else self.prob_logs_dir
        ) / self.model_dirname

    @property
    def checkpoint_dir(self) -> Path:
        """Get the checkpoint file path."""
        return (
            self.megnet_models_dir if self.meg else self.prob_models_dir
        ) / f"{self.model_dirname}-ckpt.h5"

    @property
    def test_result_path(self) -> Path:
        """Get the path to the model's final evaluation metrics."""
        return self.log_dir / "metrics.csv"

    @property
    def tb_callback(self) -> TensorBoard:
        """Get a configured TensorBoard callback."""
        tensorboard_run_loc = self.log_dir / str(
            datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
        )
        # These parameters avoid segmentation fault
        return TensorBoard(tensorboard_run_loc, write_graph=False, profile_batch=0)

    @property
    def val_prob_metrics_dir(self) -> Path:
        """Get the path to this run's validation metrics."""
        return (
            self.log_dir / "val-metrics-"
            + str(datetime.now().strftime("%Y.%m.%d.%H.%M.%S"))
            + ".csv"
        )
