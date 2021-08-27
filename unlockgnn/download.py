"""Utilities for downloading pre-trained models and data."""
import tarfile
from io import BytesIO
from os import PathLike
from pathlib import Path
from typing import Union

try:
    from typing import Literal
except ImportError:
    from typish import Literal

import pandas as pd
import requests
from .model import MEGNetProbModel

DEFAULT_MODEL_PATH = Path(__file__).parent / "models"
DEFAULT_DATA_PATH = Path(__file__).parent / "data"
MODELS_URL: str = (
    "https://github.com/a-ws-m/unlockGNN/raw/{branch}/models/{fname}.tar.gz"
)
DATA_URL: str = "https://github.com/a-ws-m/unlockGNN/raw/{branch}/data/{fname}.pkl"
AVAILABLE_MODELS = Literal["binary_e_form"]
AVAILABLE_DATA = Literal["binary_e_form"]


def _download_file(
    fname: Union[AVAILABLE_MODELS, AVAILABLE_DATA],
    branch: str,
    save_dir: PathLike,
    type: Literal["model", "data"],
) -> Path:
    """Download a file to disk if it does not already exist.

    Args:
        fname: The file name.
        branch: Which branch of the unlockGNN repository to download from.
        save_dir: The directory to check for already-downloaded models and
            in which to save newly downloaded models.
        type: The type of file.

    Returns:
        The path to the downloaded file/folder.

    """
    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir()

    specific_dir = save_dir / (f"{fname}-{branch}" + (".pkl" if type == "data" else ""))
    # Add .pkl extension only if we're downloading data
    url = MODELS_URL if type == "model" else DATA_URL
    download_url = url.format(branch=branch, fname=fname)

    if not specific_dir.exists():
        r = requests.get(download_url)
        if type == "model":
            tar_f = tarfile.open(fileobj=BytesIO(r.content))
            tar_f.extractall(specific_dir)
            tar_f.close()
        else:
            specific_dir.write_bytes(r.content)

    return specific_dir


def load_pretrained(
    model_name: AVAILABLE_MODELS,
    branch: str = "master",
    save_dir: PathLike = DEFAULT_MODEL_PATH,
) -> MEGNetProbModel:
    """Download a pre-trained model.

    Args:
        model_name: The name of the model to download.
        branch: Which branch of the unlockGNN repository to download from.
        save_dir: The directory to check for already-downloaded models and
            in which to save newly downloaded models.

    Returns:
        The downloaded model.

    """
    model_dir = _download_file(model_name, branch, save_dir, "model")
    return MEGNetProbModel.load(model_dir / model_name)


def load_data(
    data_name: AVAILABLE_DATA,
    branch: str = "master",
    save_dir: PathLike = DEFAULT_DATA_PATH,
):
    """Download sample data.

    Args:
        data_name: The name of the data to download.
        branch: Which branch of the unlockGNN repository to download from.
        save_dir: The directory to check for already-downloaded data and
            in which to save newly downloaded data.

    Returns:
        The downloaded data.

    """
    data_dir = _download_file(data_name, branch, save_dir, "data")
    return pd.read_pickle(data_dir)
