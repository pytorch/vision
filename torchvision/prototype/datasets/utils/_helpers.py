import io
from typing import Any, Tuple

import numpy as np
import PIL.Image

__all__ = ["read_mat", "image_buffer_from_array"]


def read_mat(file: io.BufferedIOBase, **kwargs: Any) -> Any:
    try:
        import scipy.io as sio
    except ImportError as error:
        raise ModuleNotFoundError(
            "Package `scipy` is required to be installed to read .mat files."
        ) from error

    return sio.loadmat(file, **kwargs)


def image_buffer_from_array(array: np.array, *, format: str) -> Tuple[str, io.BytesIO]:
    image = PIL.Image.fromarray(array)
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    return f"tmp.{format}", buffer
