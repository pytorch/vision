import warnings
from typing import Any, Dict, List, Union

import numpy as np
import PIL.Image
import torch

from torchvision.prototype import datapoints
from torchvision.prototype.transforms import Transform
from torchvision.transforms import functional as _F
from typing_extensions import Literal

from ._transform import _RandomApplyTransform
from .utils import is_simple_tensor, query_chw


class ToTensor(Transform):
    _transformed_types = (PIL.Image.Image, np.ndarray)

    def __init__(self) -> None:
        warnings.warn(
            "The transform `ToTensor()` is deprecated and will be removed in a future release. "
            "Instead, please use `transforms.Compose([transforms.ToImageTensor(), transforms.ConvertImageDtype()])`."
        )
        super().__init__()

    def _transform(self, inpt: Union[PIL.Image.Image, np.ndarray], params: Dict[str, Any]) -> torch.Tensor:
        return _F.to_tensor(inpt)


class Grayscale(Transform):
    _transformed_types = (
        datapoints.Image,
        PIL.Image.Image,
        is_simple_tensor,
        datapoints.Video,
    )

    def __init__(self, num_output_channels: Literal[1, 3] = 1) -> None:
        super().__init__()
        self.num_output_channels = num_output_channels

    def _transform(
        self, inpt: Union[datapoints.ImageType, datapoints.VideoType], params: Dict[str, Any]
    ) -> Union[datapoints.ImageType, datapoints.VideoType]:
        output = _F.rgb_to_grayscale(inpt, num_output_channels=self.num_output_channels)
        if isinstance(inpt, (datapoints.Image, datapoints.Video)):
            # TODO: Q: is the wrapping still needed? Is the type ignore still needed?
            output = inpt.wrap_like(inpt, output)  # type: ignore[arg-type]
        return output


class RandomGrayscale(_RandomApplyTransform):
    _transformed_types = (
        datapoints.Image,
        PIL.Image.Image,
        is_simple_tensor,
        datapoints.Video,
    )

    def __init__(self, p: float = 0.1) -> None:
        super().__init__(p=p)

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        num_input_channels, *_ = query_chw(flat_inputs)
        return dict(num_input_channels=num_input_channels)

    def _transform(
        self, inpt: Union[datapoints.ImageType, datapoints.VideoType], params: Dict[str, Any]
    ) -> Union[datapoints.ImageType, datapoints.VideoType]:
        output = _F.rgb_to_grayscale(inpt, num_output_channels=params["num_input_channels"])
        if isinstance(inpt, (datapoints.Image, datapoints.Video)):
            # TODO: Same as the other TODO above
            output = inpt.wrap_like(inpt, output)  # type: ignore[arg-type]
        return output
