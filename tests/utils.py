"""Testing suite shared functionality."""
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from distutils import dir_util
from math import floor
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import pytest

SplitData = Tuple[Tuple[list, list], Tuple[list, list]]

@pytest.fixture
def datadir(tmpdir, request) -> Path:
    """Access data directory.

    Fixture responsible for searching a folder with the same name of test
    module and, if available, moving all contents to a temporary directory so
    tests can use them freely.

    Source: https://stackoverflow.com/a/29631801/

    """
    filename = request.module.__file__
    test_dir, _ = os.path.splitext(filename)

    if os.path.isdir(test_dir):
        dir_util.copy_tree(test_dir, str(tmpdir))

    return tmpdir


def weights_equal(weights_a: List[np.ndarray], weights_b: List[np.ndarray]) -> bool:
    """Check equality between weights."""
    return all(
        weight1 == pytest.approx(weight2, rel=1e-6) for weight1, weight2 in zip(weights_a, weights_b)
    )

def load_df_head(fname: Path, num_entries: int=100) -> pd.DataFrame:
    """Load first entries of a pandas DataFrame in a backwards-compatible way.
    
    Args:
        fname: The pickle file to open.
        num_entries: How many values to read.
    
    """
    try:
        return pd.read_pickle(fname)[:num_entries]
    except ValueError:
        # Older python version
        import pickle5 as pkl

        with fname.open("rb") as f:
            return pkl.load(f)[:num_entries]

def train_test_split(
    structures: list, targets: list, train_frac: float = 0.8
) -> SplitData:
    """Split structures and targets into training and testing subsets."""
    num_train = floor(len(structures) * train_frac)
    return (
        (structures[:num_train], targets[:num_train]),
        (structures[num_train:], targets[num_train:]),
    )
