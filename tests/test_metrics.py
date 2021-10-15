
"""Test metrics."""
import os
from typing import Callable, List, NamedTuple

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pytest
from unlocknn.megnet_utils import Targets
from unlocknn.metrics import neg_log_likelihood, MAE, MSE, RMSE, variation, sharpness

class ToyData(NamedTuple):
    """Container for some example data."""

    predictions: Targets
    stddevs: Targets
    true_vals: Targets

class ExpectedMetrics(NamedTuple):
    """Container for expected values for metrics."""

    nll: float
    mae: float
    mse: float
    rmse: float
    sharpness: float
    variation: float

# This list should match the order for ExpectedMetrics
METRICS_TEST_ORDER: List[Callable[[Targets, Targets, Targets], float]] = [
    neg_log_likelihood,
    MAE,
    MSE,
    RMSE,
    sharpness,
    variation
]

TEST_DATA: List[ToyData] = [
    ToyData([0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [0.0, 1.0, 2.0]),
    ToyData([0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [-1.0, 1.0, 2.0]),
    ToyData([0.0, 1.0, 2.0], [1.0, 1.0, 3.0], [1.0, 1.0, 2.0]),
]

# TODO: WIP Metrics values!
TEST_EXPECTED: List[ExpectedMetrics] = [
    ExpectedMetrics(-0.731394, 0.0, 0.0, 0.0, 2.1602),
    ExpectedMetrics(-0.574423),
    ExpectedMetrics(-0.773894),
]


@pytest.mark.parametrize("toy_data,expected", zip(TEST_DATA, TEST_EXPECTED))
def test_metrics(toy_data: ToyData, expected: ExpectedMetrics):
    """Test that the metrics for the toy data match expected values."""
    for metric, expected_value in zip(METRICS_TEST_ORDER, expected):
        assert metric(*toy_data) == pytest.approx(expected_value)
