import os
import pathlib
from typing import Optional, Union

from torch.hub import _get_torch_home

HOME = pathlib.Path(_get_torch_home()) / "datasets" / "vision"


def home(root: Optional[Union[str, pathlib.Path]] = None) -> pathlib.Path:
    global HOME
    if root is not None:
        HOME = pathlib.Path(root).expanduser().resolve()
        return HOME

    root = os.getenv("TORCHVISION_DATASETS_HOME")
    if root is not None:
        return pathlib.Path(root)

    return HOME
