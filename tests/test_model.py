"""Test model features."""
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import random as python_random
from distutils import dir_util
from math import floor
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from megnet.models import MEGNetModel
from unlocknn import MEGNetProbModel
from unlocknn.initializers import SampleInitializer

np.random.seed(123)
python_random.seed(123)
tf.random.set_seed(123)


@pytest.fixture
def datadir(tmpdir, request):
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
        (weight1 == weight2).all() for weight1, weight2 in zip(weights_a, weights_b)
    )


def train_test_split(
    structures: list, targets: list, train_frac: float = 0.8
) -> Tuple[Tuple[list, list], Tuple[list, list]]:
    """Split structures and targets into training and testing subsets."""
    num_train = floor(len(structures) * train_frac)
    return (
        (structures[:num_train], targets[:num_train]),
        (structures[num_train:], targets[num_train:]),
    )


def test_sample_init(datadir: Path):
    """Test the SampleInitializer."""
    megnet_e_form_model = MEGNetModel.from_file(str(datadir / "formation_energy.hdf5"))
    binary_dir = datadir / "mp_binary_on_hull.pkl"

    try:
        binary_df = pd.read_pickle(binary_dir)[:100]
    except ValueError:
        # Older python version
        import pickle5 as pkl

        with binary_dir.open("rb") as f:
            binary_df = pkl.load(f)

    structures = binary_df["structure"].tolist()
    formation_energies = binary_df["formation_energy_per_atom"].tolist()

    (train_structs, _), (_, _) = train_test_split(structures, formation_energies)
    initializer = SampleInitializer(train_structs, megnet_e_form_model, batch_size=32)
    MEGNetProbModel(megnet_e_form_model, 10, index_initializer=initializer)
    # If this works without any errors, we're doing OK


@pytest.mark.parametrize("use_norm", [True, False])
def test_meg_prob(tmp_path: Path, datadir: Path, use_norm: bool):
    """Test creation, training and I/O of a `MEGNetProbModel`."""
    save_dir = tmp_path / ("norm_model" if use_norm else "unnorm_model")
    ckpt_path = tmp_path / "checkpoint.h5"
    megnet_e_form_model = MEGNetModel.from_file(str(datadir / "formation_energy.hdf5"))
    binary_dir = datadir / "mp_binary_on_hull.pkl"

    try:
        binary_df = pd.read_pickle(binary_dir)[:100]
    except ValueError:
        # Older python version
        import pickle5 as pkl

        with binary_dir.open("rb") as f:
            binary_df = pkl.load(f)

    structures = binary_df["structure"].tolist()
    formation_energies = binary_df["formation_energy_per_atom"].tolist()

    (train_structs, train_targets), (test_structs, test_targets) = train_test_split(
        structures, formation_energies
    )
    prob_model = MEGNetProbModel(megnet_e_form_model, 10, use_normalization=use_norm)

    # Test weights equality
    last_nn_idx = -2 if use_norm else -1
    meg_nn_weights = [
        layer.get_weights() for layer in megnet_e_form_model.model.layers[:-1]
    ]
    prob_model_nn_weights = [
        layer.get_weights() for layer in prob_model.model.layers[:last_nn_idx]
    ]
    for meg_layer, prob_layer in zip(meg_nn_weights, prob_model_nn_weights):
        assert weights_equal(meg_layer, prob_layer)

    init_performance = prob_model.evaluate(test_structs, test_targets)
    init_loss = init_performance["loss"]

    # Test training without validation
    prob_model.train(
        train_structs, train_targets, 1, batch_size=32, ckpt_path=ckpt_path
    )
    # Test training with validation
    prob_model.train(
        train_structs,
        train_targets,
        1,
        test_structs,
        test_targets,
        batch_size=32,
        ckpt_path=ckpt_path,
    )

    # Save and reload model from disk
    prob_model.save(save_dir, ckpt_path=ckpt_path)
    loaded_model = MEGNetProbModel.load(save_dir, load_ckpt=False)
    assert weights_equal(
        prob_model.model.get_weights(), loaded_model.model.get_weights()
    )

    # Test prediction
    prob_model.predict(train_structs, batch_size=32)
