"""Utilities for object serialization."""
import typing
from io import BytesIO

import numpy as np


def serialize_array(arr: np.ndarray) -> bytes:
    """Serialize an array into a binary format.

    The array is serialized into `numpy`'s NPY format.
    It can be retrieved using

    Args:
        arr (:obj:`np.ndarray`): The array to serialize.

    Returns:
        serialized (bytes): The serialized array.

    """
    outfile = BytesIO()
    np.save(outfile, arr)
    serialized = outfile.getvalue()
    outfile.close()
    return serialized


def deserialize_array(ser: bytes) -> np.ndarray:
    """Deserialize an array.

    Args:
        ser (bytes): The serialized array, in `numpy`
            NPY format.

    Returns:
        arr (:obj:`np.ndarray`): The deserialized array.

    """
    infile = BytesIO(ser)
    arr = np.load(infile)
    infile.close()
    return arr
