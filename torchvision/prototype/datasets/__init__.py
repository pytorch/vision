try:
    import torchdata
except (ModuleNotFoundError, TypeError) as error:
    raise ModuleNotFoundError(
        "`torchvision.prototype.datasets` depends on PyTorch's `torchdata` (https://github.com/pytorch/data). "
        "You can install it with `pip install git+https://github.com/pytorch/data.git`. "
        "Note that you cannot install it with `pip install torchdata`, since this is another package."
    ) from error


from ._home import home
from . import decoder, utils

# Load this last, since some parts depend on the above being loaded first
from ._api import register, list, info, load
from ._folder import from_data_folder, from_image_folder
