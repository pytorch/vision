import os
from typing import Optional

import torchvision._internally_replaced_utils as _iru


def home(root: Optional[str] = None) -> str:
    if root is not None:
        _iru._HOME = root
        return _iru._HOME

    root = os.getenv("TORCHVISION_DATASETS_HOME")
    if root is not None:
        return root

    return _iru._HOME


def use_sharded_dataset(use: Optional[bool] = None) -> bool:
    if use is not None:
        _iru._USE_SHARDED_DATASETS = use
        return _iru._USE_SHARDED_DATASETS

    use = os.getenv("TORCHVISION_SHARDED_DATASETS")
    if use is not None:
        return use == "1"

    return _iru._USE_SHARDED_DATASETS
