"""Test model features."""
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import random as python_random
from pathlib import Path
from typing import List

import numpy as np
import pytest
import tensorflow as tf
from megnet.models import MEGNetModel
from pymatgen.core.structure import Structure
from unlocknn import MEGNetProbModel
from unlocknn.initializers import SampleInitializer
from unlocknn.megnet_utils import MEGNetGraph

from .utils import (SplitData, datadir, load_df_head, train_test_split,
                   weights_equal)

np.random.seed(123)
python_random.seed(123)
tf.random.set_seed(123)


def structs_to_graphs(meg_model: MEGNetModel, structures: List[Structure]) -> List[MEGNetGraph]:
    """Convert Structures to graphs."""
    dummy_targets = [0.0] * len(structures)
    return meg_model.get_all_graphs_targets(structures, dummy_targets)[0]


@pytest.fixture
def split_data(datadir: Path) -> SplitData:
    """Get some example data, split into training and test subsets.
    
    Returns:
        train_structs
        
        train_targets
        
        test_structs
        
        test_targets
    
    """
    binary_dir = datadir / "mp_binary_on_hull.pkl"
    binary_df = load_df_head(binary_dir)
    structures = binary_df["structure"].tolist()
    formation_energies = binary_df["formation_energy_per_atom"].tolist()

    return train_test_split(
        structures, formation_energies
    )


def test_sample_init(datadir: Path, split_data: SplitData):
    """Test the SampleInitializer."""
    megnet_e_form_model = MEGNetModel.from_file(str(datadir / "formation_energy.hdf5"))
    (train_structs, _), (_, _) = split_data

    initializer = SampleInitializer(train_structs, megnet_e_form_model, batch_size=32)
    MEGNetProbModel(megnet_e_form_model, 10, index_initializer=initializer)
    # If this works without any errors, we're doing OK


@pytest.mark.parametrize("use_norm", [True, False])
def test_model_reload(tmp_path: Path, datadir: Path, use_norm: bool):
    """Test saving and reloading a model from disk.
    
    Asserts weight equality.
    
    """
    save_dir = tmp_path / ("norm_model" if use_norm else "unnorm_model")
    megnet_e_form_model = MEGNetModel.from_file(str(datadir / "formation_energy.hdf5"))
    prob_model = MEGNetProbModel(megnet_e_form_model, 10, use_normalization=use_norm)

    prob_model.save(save_dir, ckpt_path=None)
    loaded_model = MEGNetProbModel.load(save_dir, load_ckpt=False)
    assert weights_equal(
        prob_model.model.get_weights(), loaded_model.model.get_weights()
    )

@pytest.mark.parametrize("use_graphs", [True, False])
@pytest.mark.parametrize("use_norm", [True, False])
def test_model_prediction(datadir: Path, split_data: SplitData, use_norm: bool, use_graphs: bool):
    """Test that model prediction has expected dimensions."""
    model_name = "prob_e_form_" + ("" if use_norm else "un") + "norm"
    prob_model = MEGNetProbModel.load(datadir / model_name, load_ckpt=False)
    (train_structs, _), (_, _) = split_data
    train_input = structs_to_graphs(prob_model.meg_model, train_structs) if use_graphs else train_structs

    prob_model.update_pred_model()

    prediction, stddev = prob_model.predict(train_input, batch_size=16)

    # Arrays should have the same shape: flat, with one entry per structure
    expected_shape = (len(train_structs),)
    assert prediction.shape == expected_shape
    assert stddev.shape == expected_shape


@pytest.mark.parametrize("use_graphs", [True, False])
@pytest.mark.parametrize("use_norm", [True, False])
def test_model_training(tmp_path: Path, datadir: Path, split_data: SplitData, use_norm: bool, use_graphs: bool):
    """Test training then saving a model.
    
    Check if expected weights update/are frozen. Saves model after training to
    check that checkpoints are handled correctly.

    """
    save_dir = tmp_path / ("norm_model" if use_norm else "unnorm_model")
    ckpt_path = tmp_path / "checkpoint.h5"
    model_name = "prob_e_form_{}norm".format("" if use_norm else "un")
    (train_structs, train_targets), (test_structs, test_targets) = split_data
    last_nn_idx = -2 if use_norm else -1

    prob_model = MEGNetProbModel.load(datadir / model_name, load_ckpt=False)
    train_input = structs_to_graphs(prob_model.meg_model, train_structs) if use_graphs else train_structs
    test_input = structs_to_graphs(prob_model.meg_model, test_structs) if use_graphs else test_structs

    for layer in prob_model.model.layers:
        print(layer.name)

    init_weights = [layer.get_weights() for layer in prob_model.model.layers]
    init_nn_weights = init_weights[:last_nn_idx]
    init_uq_weights = init_weights[last_nn_idx:]

    # For initial training, we expect the weights before the
    # `Norm`/`VGP` layers to be frozen initially, so these shouldn't change.
    # The rest should, as the model isn't optimised.
    prob_model.train(
        train_input, train_targets, 1, test_input, test_targets, batch_size=32, ckpt_path=ckpt_path
    )

    final_weights = [layer.get_weights() for layer in prob_model.model.layers]
    final_nn_weights = final_weights[:last_nn_idx]
    final_uq_weights = final_weights[last_nn_idx:]

    for init_nn_weight, final_nn_weight in zip(init_nn_weights, final_nn_weights):
        assert weights_equal(init_nn_weight, final_nn_weight)

    for init_uq_weight, final_uq_weight in zip(init_uq_weights, final_uq_weights):
        assert not weights_equal(init_uq_weight, final_uq_weight)

    # * Test saving the model
    # If this runs without errors, we're OK.
    prob_model.save(save_dir, ckpt_path=ckpt_path)

@pytest.mark.parametrize("use_norm", [True, False])
def test_meg_weights_preserved(datadir: Path, use_norm: bool):
    """Test that the MEGNet weights are correctly transferred to a MEGNetProbModel."""
    megnet_e_form_model = MEGNetModel.from_file(str(datadir / "formation_energy.hdf5"))

    # * Initialize model
    prob_model = MEGNetProbModel(megnet_e_form_model, 10, use_normalization=use_norm)

    # * Test weights equality
    # Ensure that the weights of the model that we copied over are unchanged
    # and in the correct layers.
    last_nn_idx = -2 if use_norm else -1
    meg_nn_weights = [
        layer.get_weights() for layer in megnet_e_form_model.model.layers[:-1]
    ]
    prob_model_nn_weights = [
        layer.get_weights() for layer in prob_model.model.layers[:last_nn_idx]
    ]
    for meg_layer, prob_layer in zip(meg_nn_weights, prob_model_nn_weights):
        assert weights_equal(meg_layer, prob_layer)

@pytest.mark.parametrize("use_norm", [True, False])
def test_freezing(datadir: Path, use_norm: bool):
    """Test that model layers can be frozen and thawed."""
    model_name = "prob_e_form_" + ("" if use_norm else "un") + "norm"
    prob_model = MEGNetProbModel.load(datadir / model_name, load_ckpt=False)
    
    # Loaded model should have NN frozen and VGP and Norm unfrozen
    assert prob_model.nn_frozen
    assert not prob_model.vgp_frozen
    if use_norm:
        assert prob_model.norm_frozen is False
    else:
        assert prob_model.norm_frozen is None
    
    # Try freezing/thawing
    prob_model.set_frozen("VGP")
    assert prob_model.vgp_frozen

    # Test thawing with a single item list
    prob_model.set_frozen(["VGP"], False)
    assert not prob_model.vgp_frozen

    # Test error handling
    if use_norm:
        prob_model.set_frozen("Norm")
        assert prob_model.norm_frozen
    else:
        with pytest.raises(ValueError):
            prob_model.set_frozen("Norm")

    # Test thawing with a list
    prob_model.set_frozen(["NN", "VGP"], False)
    assert not prob_model.nn_frozen
    assert not prob_model.vgp_frozen
