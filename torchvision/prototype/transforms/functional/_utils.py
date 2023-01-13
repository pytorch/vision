from typing import Any

import torch
from torchvision.prototype.datapoints._datapoint import Datapoint

from typing_extensions import TypeGuard


def is_simple_tensor(inpt: Any) -> TypeGuard[torch.Tensor]:
    return isinstance(inpt, torch.Tensor) and not isinstance(inpt, Datapoint)
