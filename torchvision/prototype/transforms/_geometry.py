from typing import Any, Dict, List, Optional, Sequence, Type, Union

import PIL.Image
import torch

from torchvision import tv_tensors
from torchvision.prototype.tv_tensors import Label, OneHotLabel
from torchvision.transforms.v2 import functional as F, Transform
from torchvision.transforms.v2._utils import (
    _FillType,
    _get_fill,
    _setup_fill_arg,
    _setup_size,
    get_bounding_boxes,
    has_any,
    is_pure_tensor,
    query_size,
)


class FixedSizeCrop(Transform):
    def __init__(
        self,
        size: Union[int, Sequence[int]],
        fill: Union[_FillType, Dict[Union[Type, str], _FillType]] = 0,
        padding_mode: str = "constant",
    ) -> None:
        super().__init__()
        size = tuple(_setup_size(size, error_msg="Please provide only two dimensions (h, w) for size."))
        self.crop_height = size[0]
        self.crop_width = size[1]

        self.fill = fill
        self._fill = _setup_fill_arg(fill)

        self.padding_mode = padding_mode

    def check_inputs(self, flat_inputs: List[Any]) -> None:
        if not has_any(
            flat_inputs,
            PIL.Image.Image,
            tv_tensors.Image,
            is_pure_tensor,
            tv_tensors.Video,
        ):
            raise TypeError(
                f"{type(self).__name__}() requires input sample to contain an tensor or PIL image or a Video."
            )

        if has_any(flat_inputs, tv_tensors.BoundingBoxes) and not has_any(flat_inputs, Label, OneHotLabel):
            raise TypeError(
                f"If a BoundingBoxes is contained in the input sample, "
                f"{type(self).__name__}() also requires it to contain a Label or OneHotLabel."
            )

    def make_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        height, width = query_size(flat_inputs)
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
            bounding_boxes = get_bounding_boxes(flat_inputs)
        except ValueError:
            bounding_boxes = None

        if needs_crop and bounding_boxes is not None:
            format = bounding_boxes.format
            bounding_boxes, canvas_size = F.crop_bounding_boxes(
                bounding_boxes.as_subclass(torch.Tensor),
                format=format,
                top=top,
                left=left,
                height=new_height,
                width=new_width,
            )
            bounding_boxes = F.clamp_bounding_boxes(bounding_boxes, format=format, canvas_size=canvas_size)
            height_and_width = F.convert_bounding_box_format(
                bounding_boxes, old_format=format, new_format=tv_tensors.BoundingBoxFormat.XYWH
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

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if params["needs_crop"]:
            inpt = self._call_kernel(
                F.crop,
                inpt,
                top=params["top"],
                left=params["left"],
                height=params["height"],
                width=params["width"],
            )

        if params["is_valid"] is not None:
            if isinstance(inpt, (Label, OneHotLabel, tv_tensors.Mask)):
                inpt = tv_tensors.wrap(inpt[params["is_valid"]], like=inpt)
            elif isinstance(inpt, tv_tensors.BoundingBoxes):
                inpt = tv_tensors.wrap(
                    F.clamp_bounding_boxes(inpt[params["is_valid"]], format=inpt.format, canvas_size=inpt.canvas_size),
                    like=inpt,
                )

        if params["needs_pad"]:
            fill = _get_fill(self._fill, type(inpt))
            inpt = self._call_kernel(F.pad, inpt, params["padding"], fill=fill, padding_mode=self.padding_mode)

        return inpt
