"""Utilities for downloading pre-trained models and data."""
import tarfile
from io import BytesIO
from os import PathLike
from pathlib import Path
from typing import Union

try:
    from typing import Literal
except ImportError:  # pragma: no cover
    from typish import Literal

import pandas as pd
import requests
from pymatgen.core.structure import Structure

from .model import MEGNetProbModel

DEFAULT_MODEL_PATH = Path(__file__).parent / "models"
DEFAULT_DATA_PATH = Path(__file__).parent / "data"
MODELS_URL: str = (
    "https://github.com/a-ws-m/unlockNN/raw/{branch}/models/{fname}.tar.gz"
)
DATA_URL: str = "https://github.com/a-ws-m/unlockNN/raw/{branch}/data/{fname}.parquet"
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
        branch: Which branch of the unlockNN repository to download from.
        save_dir: The directory to check for already-downloaded models and
            in which to save newly downloaded models.
        type: The type of file.

    Returns:
        The path to the downloaded file/folder.

    """
    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir()

    specific_dir = save_dir / (
        f"{fname}-{branch}" + (".parquet" if type == "data" else "")
    )
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

    A list of available models and their descriptions can be
    found at https://github.com/a-ws-m/unlockNN/tree/master/models.

    Args:
        model_name: The name of the model to download.
        branch: Which branch of the unlockNN repository to download from.
        save_dir: The directory to check for already-downloaded models and
            in which to save newly downloaded models.

    Returns:
        The downloaded model.

    Examples:
        Download model for predicting binary compounds' formation energies:

        >>> model = load_pretrained("binary_e_form")

    """
    model_dir = _download_file(model_name, branch, save_dir, "model")
    return MEGNetProbModel.load(model_dir / model_name)


def load_data(
    data_name: AVAILABLE_DATA,
    branch: str = "master",
    save_dir: PathLike = DEFAULT_DATA_PATH,
):
    """Download sample data.

    A list of available data, their sources and descriptions can be
    found at https://github.com/a-ws-m/unlockNN/tree/master/data.

    Args:
        data_name: The name of the data to download.
        branch: Which branch of the unlockNN repository to download from.
        save_dir: The directory to check for already-downloaded data and
            in which to save newly downloaded data.

    Returns:
        The downloaded data.

    Examples:
        Download binary compounds and their formation energies, then print
        the first dataset entry:

        >>> data = load_data("binary_e_form")
        >>> print(data.iloc[0])
        structure                    [[ 1.982598   -4.08421341  3.2051745 ] La, [1....
        formation_energy_per_atom                                            -0.737439
        Name: 0, dtype: object

    """
    data_dir = _download_file(data_name, branch, save_dir, "data")
    return _load_struct_data(data_dir)


def _load_struct_data(fname: PathLike) -> pd.DataFrame:
    """Load data containing ``Structure``s from a file.

    Deserializes the "structure" column to a string.
    The converse of :func:`save_struct_data`.

    Args:
        fname: Whence to load the file.

    Returns:
        The deserialized ``DataFrame``.

    """
    serial_df = pd.read_parquet(fname)
    serial_df["structure"] = serial_df["structure"].map(
        lambda string: Structure.from_str(string, "json")
    )
    return serial_df


def save_struct_data(df: pd.DataFrame, fname: PathLike):
    """Save data containing ``Structure`` s to a file.

    Serializes the "structure" column to a string.
    The converse of :func:`_load_struct_data`.

    Args:
        df: The :class:`pd.DataFrame` to serialize.
        fname: Where to save the file.

    """
    serial_df = df.copy()
    serial_df["structure"] = serial_df["structure"].map(
        lambda struct: struct.to("json")
    )
    serial_df.to_parquet(fname)


if __name__ == "__main__":  # pragma: no cover
    import doctest

    doctest.testmod()
