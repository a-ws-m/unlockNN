"""Boilerplate script for training and evaluation on matbench datasets."""
import gzip
import json
from math import floor
from os import mkdir
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import requests
from megnet.models import MEGNetModel
from pymatgen.core.structure import Structure
from sklearn.model_selection import KFold
from unlocknn.megnet_utils import default_megnet_config, create_megnet_input

from .. import utils


def download_data(url: str, save_dir: Path) -> pd.DataFrame:
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


def convert_to_graphs(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Structures in a DataFrame to graphs."""
    dummy_model = MEGNetModel(**default_megnet_config())
    _, df["graph"] = create_megnet_input(
        dummy_model, df["structure"].values, shuffle=False
    )
    return df


class MatbenchTrainer(utils.UnlockTrainer):
    """Training suite for matbench regression benchmarks."""

    def __init__(
        self,
        data_name: str,
        data_url: str,
        target_col: str,
        batch_size: int = 32,
    ) -> None:
        """Initialize training suite."""
        self.data_name = data_name
        self.data_url = data_url
        self.target_col = target_col
        super().__init__(batch_size=batch_size)

    def load_data(self) -> utils.Dataset:
        """Load matbench data."""
        if self.graph_data_file.exists():
            df = pd.read_pickle(self.graph_data_file)
        else:
            if self.raw_data_file.exists():
                raw_df = pd.read_pickle(self.raw_data_file)
            else:
                raw_df = download_data(self.data_url, self.raw_data_file)
            df = convert_to_graphs(raw_df)
            df.to_pickle(self.graph_data_file)

        # 5-fold splitting per matbench rules
        kf = KFold(n_splits=5, shuffle=True, random_state=18012019)
        train_indexes, testing_indexes = list(kf.split(df))[self.fold]

        # Train indexes must be further divided into training and validation
        num_train_samples = len(train_indexes)
        num_training = floor(0.95 * num_train_samples)

        training_indexes = train_indexes[:num_training]
        val_indexes = train_indexes[num_training:]

        print(f"Number of training samples: {len(training_indexes)}")
        print(f"Number of validation samples: {len(val_indexes)}")
        print(
            f"Training ratio: {len(training_indexes) / (len(val_indexes) + len(training_indexes))}"
        )

        return utils.Dataset(
            train_input=df["graph"].iloc[training_indexes].values,
            train_targets=df[self.target_col].iloc[training_indexes].values,
            val_input=df["graph"].iloc[val_indexes].values,
            val_targets=df[self.target_col].iloc[val_indexes].values,
            test_input=df["graph"].iloc[testing_indexes].values,
            test_targets=df[self.target_col].iloc[testing_indexes].values,
        )

    @property
    def task_name(self) -> str:
        return self.data_name

    def create_dir_structure(self):
        super().create_dir_structure()
        self.data_dir = self.root_dir / "data"
        if not self.data_dir.exists():
            mkdir(self.data_dir)

        self.raw_data_file = self.data_dir / f"{self.data_name}.pkl"
        self.graph_data_file = self.data_dir / f"{self.data_name}_graphs.pkl"

    @property
    def num_folds(self) -> int:
        return 5
