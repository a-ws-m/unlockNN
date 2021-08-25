"""Utilities for downloading pre-trained models and data."""
import tarfile
from io import BytesIO
from os import PathLike
from pathlib import Path

try:
    from typing import Literal
except ImportError:
    from typish import Literal

import requests
from .model import MEGNetProbModel

DEFAULT_MODEL_PATH = Path(__file__).parent / "models"
MODELS_URL: str = (
    "https://github.com/a-ws-m/unlockGNN/raw/{branch}/models/{model}.tar.gz"
)
AVAILABLE_MODEL = Literal["binary_e_form"]


def load_pretrained(
    model_name: AVAILABLE_MODEL,
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
    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir()

    model_dir = save_dir / f"{model_name}-{branch}"
    download_url = MODELS_URL.format(branch=branch, model=model_name)

    if not model_dir.exists():
        r = requests.get(download_url)
        tf = tarfile.open(fileobj=BytesIO(r.content))
        tf.extractall(model_dir)
        tf.close()

    return MEGNetProbModel.load(model_dir / model_name)
