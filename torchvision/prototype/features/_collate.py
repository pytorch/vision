from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Type, Union

from torch.utils.data._utils.collate import collate, collate_tensor_fn, default_collate_fn_map
from torchvision.prototype.features import Image, Label, Mask, OneHotLabel


def no_collate_fn(
    batch: Sequence[Any], *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None
) -> Any:
    return batch


def new_like_collate_fn(
    batch: Sequence[Any], *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None
) -> Any:
    feature = batch[0]
    tensor = collate_tensor_fn(batch, collate_fn_map=collate_fn_map)
    return type(feature).new_like(feature, tensor)


vision_collate_fn_map = {
    (Image, Mask, Label, OneHotLabel): new_like_collate_fn,
    type(None): no_collate_fn,
    **default_collate_fn_map,
}


def vision_collate(batch: Sequence[Any]) -> Any:
    return collate(batch, collate_fn_map=vision_collate_fn_map)
