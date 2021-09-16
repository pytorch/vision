import os
import pathlib
from typing import Optional, Union

from torch.hub import _get_torch_home

# TODO: This needs a better default
HOME = pathlib.Path(_get_torch_home()) / "datasets" / "vision"


def home(home: Optional[Union[str, pathlib.Path]] = None) -> pathlib.Path:
    global HOME
    if home is not None:
        HOME = pathlib.Path(home).expanduser().resolve()
        return HOME

    home = os.getenv("TORCHVISION_DATASETS_HOME")
    if home is not None:
        return pathlib.Path(home)

    return HOME
