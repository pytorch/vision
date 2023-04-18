from typing import Any, Dict, List, Optional, Sequence, Type, Union

import PIL.Image
import torch

from torchvision import datapoints
from torchvision.prototype.datapoints import Label, OneHotLabel
from torchvision.transforms.v2 import functional as F, Transform
from torchvision.transforms.v2._utils import _setup_fill_arg, _setup_size
from torchvision.transforms.v2.utils import has_any, is_simple_tensor, query_bounding_box, query_spatial_size


class FixedSizeCrop(Transform):
    def __init__(
        self,
        size: Union[int, Sequence[int]],
        fill: Union[datapoints._FillType, Dict[Type, datapoints._FillType]] = 0,
        padding_mode: str = "constant",
    ) -> None:
        super().__init__()
        size = tuple(_setup_size(size, error_msg="Please provide only two dimensions (h, w) for size."))
        self.crop_height = size[0]
        self.crop_width = size[1]

        self.fill = fill
        self._fill = _setup_fill_arg(fill)

        self.padding_mode = padding_mode

    def _check_inputs(self, flat_inputs: List[Any]) -> None:
        if not has_any(
            flat_inputs,
            PIL.Image.Image,
            datapoints.Image,
            is_simple_tensor,
            datapoints.Video,
        ):
            raise TypeError(
                f"{type(self).__name__}() requires input sample to contain an tensor or PIL image or a Video."
            )

        if has_any(flat_inputs, datapoints.BoundingBox) and not has_any(flat_inputs, Label, OneHotLabel):
            raise TypeError(
                f"If a BoundingBox is contained in the input sample, "
                f"{type(self).__name__}() also requires it to contain a Label or OneHotLabel."
            )

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        height, width = query_spatial_size(flat_inputs)
        new_height = min(height, self.crop_height)
        new_width = min(width, self.crop_width)

        needs_crop = new_height != height or new_width != width

        offset_height = max(height - self.crop_height, 0)
        offset_width = max(width - self.crop_width, 0)

        r = torch.rand(1)
        top = int(offset_height * r)
        left = int(offset_width * r)

        bounding_boxes: Optional[torch.Tensor]
        try:
            bounding_boxes = query_bounding_box(flat_inputs)
        except ValueError:
            bounding_boxes = None

        if needs_crop and bounding_boxes is not None:
            format = bounding_boxes.format
            bounding_boxes, spatial_size = F.crop_bounding_box(
                bounding_boxes.as_subclass(torch.Tensor),
                format=format,
                top=top,
                left=left,
                height=new_height,
                width=new_width,
            )
            bounding_boxes = F.clamp_bounding_box(bounding_boxes, format=format, spatial_size=spatial_size)
            height_and_width = F.convert_format_bounding_box(
                bounding_boxes, old_format=format, new_format=datapoints.BoundingBoxFormat.XYWH
            )[..., 2:]
            is_valid = torch.all(height_and_width > 0, dim=-1)
        else:
            is_valid = None

        pad_bottom = max(self.crop_height - new_height, 0)
        pad_right = max(self.crop_width - new_width, 0)

        needs_pad = pad_bottom != 0 or pad_right != 0

        return dict(
            needs_crop=needs_crop,
            top=top,
            left=left,
            height=new_height,
            width=new_width,
            is_valid=is_valid,
            padding=[0, 0, pad_right, pad_bottom],
            needs_pad=needs_pad,
        )

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if params["needs_crop"]:
            inpt = F.crop(
                inpt,
                top=params["top"],
                left=params["left"],
                height=params["height"],
                width=params["width"],
            )

        if params["is_valid"] is not None:
            if isinstance(inpt, (Label, OneHotLabel, datapoints.Mask)):
                inpt = inpt.wrap_like(inpt, inpt[params["is_valid"]])  # type: ignore[arg-type]
            elif isinstance(inpt, datapoints.BoundingBox):
                inpt = datapoints.BoundingBox.wrap_like(
                    inpt,
                    F.clamp_bounding_box(inpt[params["is_valid"]], format=inpt.format, spatial_size=inpt.spatial_size),
                )

        if params["needs_pad"]:
            fill = self._fill[type(inpt)]
            inpt = F.pad(inpt, params["padding"], fill=fill, padding_mode=self.padding_mode)

        return inpt
