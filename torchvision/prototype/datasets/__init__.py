try:
    import torchdata
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "`torchvision.prototype.datasets` depends on PyTorch's `torchdata` (https://github.com/pytorch/data). "
        "You can install it with `pip install --pre torchdata --extra-index-url https://download.pytorch.org/whl/nightly/cpu"
    ) from None

from . import utils
from ._home import home

# Load this last, since some parts depend on the above being loaded first
from ._api import list_datasets, info, load, register_info, register_dataset  # usort: skip
from ._folder import from_data_folder, from_image_folder
from ._builtin import *
