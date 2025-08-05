import collections
import math
import pathlib
import warnings
from itertools import repeat
from types import FunctionType
from typing import Any, BinaryIO, Optional, Union

import numpy as np
import torch
from PIL import __version__ as PILLOW_VERSION_STRING, Image, ImageColor, ImageDraw, ImageFont


__all__ = [
    "_Image_fromarray",
    "make_grid",
    "save_image",
    "draw_bounding_boxes",
    "draw_segmentation_masks",
    "draw_keypoints",
    "flow_to_image",
]


@torch.no_grad()
def make_grid(
    tensor: Union[torch.Tensor, list[torch.Tensor]],
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = False,
    value_range: Optional[tuple[int, int]] = None,
    scale_each: bool = False,
    pad_value: float = 0.0,
) -> torch.Tensor:
    """
    Make a grid of images.

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
        padding (int, optional): amount of padding. Default: ``2``.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by the min and max values specified by ``value_range``. Default: ``False``.
        value_range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If ``True``, scale each image in the batch of
            images separately rather than the (min, max) over all images. Default: ``False``.
        pad_value (float, optional): Value for the padded pixels. Default: ``0``.

    Returns:
        grid (Tensor): the tensor containing grid of images.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(make_grid)
    if not torch.is_tensor(tensor):
        if isinstance(tensor, list):
            for t in tensor:
                if not torch.is_tensor(t):
                    raise TypeError(f"tensor or list of tensors expected, got a list containing {type(t)}")
        else:
            raise TypeError(f"tensor or list of tensors expected, got {type(tensor)}")

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if value_range is not None and not isinstance(value_range, tuple):
            raise TypeError("value_range has to be a tuple (min, max) if specified. min and max are numbers")

        def norm_ip(img, low, high):
            img.clamp_(min=low, max=high)
            img.sub_(low).div_(max(high - low, 1e-5))

        def norm_range(t, value_range):
            if value_range is not None:
                norm_ip(t, value_range[0], value_range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, value_range)
        else:
            norm_range(tensor, value_range)

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor should be of type torch.Tensor")
    if tensor.size(0) == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    num_channels = tensor.size(1)
    grid = tensor.new_full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            # Tensor.copy_() is a valid method but seems to be missing from the stubs
            # https://pytorch.org/docs/stable/tensors.html#torch.Tensor.copy_
            grid.narrow(1, y * height + padding, height - padding).narrow(  # type: ignore[attr-defined]
                2, x * width + padding, width - padding
            ).copy_(tensor[k])
            k = k + 1
    return grid


class _ImageDrawTV(ImageDraw.ImageDraw):
    """
    A wrapper around PIL.ImageDraw to add functionalities for drawing rotated bounding boxes.
    """

    def oriented_rectangle(self, xy, fill=None, outline=None, width=1):
        self.dashed_line(((xy[0], xy[1]), (xy[2], xy[3])), width=width, fill=outline)
        for i in range(2, len(xy), 2):
            self.line(
                ((xy[i], xy[i + 1]), (xy[(i + 2) % len(xy)], xy[(i + 3) % len(xy)])),
                width=width,
                fill=outline,
            )
        self.polygon(xy, fill=fill, outline=None, width=0)

    def dashed_line(self, xy, fill=None, width=0, joint=None, dash_length=5, space_length=5):
        # Calculate the total length of the line
        total_length = 0
        for i in range(1, len(xy)):
            x1, y1 = xy[i - 1]
            x2, y2 = xy[i]
            total_length += ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        # Initialize the current position and the current dash
        current_position = 0
        current_dash = True
        # Iterate over the coordinates of the line
        for i in range(1, len(xy)):
            x1, y1 = xy[i - 1]
            x2, y2 = xy[i]
            # Calculate the length of this segment
            segment_length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            # While there are still dashes to draw on this segment
            while segment_length > 0:
                # Calculate the length of this dash
                dash_length_to_draw = min(segment_length, dash_length if current_dash else space_length)
                # Calculate the end point of this dash
                dx = x2 - x1
                dy = y2 - y1
                angle = math.atan2(dy, dx)
                end_x = x1 + math.cos(angle) * dash_length_to_draw
                end_y = y1 + math.sin(angle) * dash_length_to_draw
                # If this is a dash, draw it
                if current_dash:
                    self.line([(x1, y1), (end_x, end_y)], fill, width, joint)
                # Update the current position and the current dash
                current_position += dash_length_to_draw
                segment_length -= dash_length_to_draw
                x1, y1 = end_x, end_y
                current_dash = not current_dash


def _Image_fromarray(
    obj: np.ndarray,
    mode: str,
) -> Image.Image:
    """
    A wrapper around PIL.Image.fromarray to mitigate the deprecation of the
    mode paramter. See:
      https://pillow.readthedocs.io/en/stable/releasenotes/11.3.0.html#image-fromarray-mode-parameter
    """

    # This may throw if the version string is from an install that comes from a
    # non-stable or development version. We'll fall back to the old behavior in
    # such cases.
    try:
        PILLOW_VERSION = tuple(int(x) for x in PILLOW_VERSION_STRING.split("."))
    except Exception:
        PILLOW_VERSION = None

    if PILLOW_VERSION is not None and PILLOW_VERSION >= (11, 3):
        # The actual PR that implements the deprecation has more context for why
        # it was done, and also points out some problems:
        #
        #    https://github.com/python-pillow/Pillow/pull/9018
        #
        # Our use case falls into those problems. We actually rely on the old
        # behavior of Image.fromarray():
        #
        #    new behavior: PIL will infer the image mode from the data passed
        #                  in. That is, the type and shape determines the mode.
        #
        #    old behiavor: The mode will change how PIL reads the image,
        #                  regardless of the data. That is, it will make the
        #                  data work with the mode.
        #
        # Our uses of Image.fromarray() are effectively a "turn into PIL image
        # AND convert the kind" operation. In particular, in
        # functional.to_pil_image() and transforms.ToPILImage.
        #
        # However, Image.frombuffer() still performs this conversion. The code
        # below is lifted from the new implementation of Image.fromarray(). We
        # omit the code that infers the mode, and use the code that figures out
        # from the data passed in (obj) what the correct parameters are to
        # Image.frombuffer().
        #
        # Note that the alternate solution below does not work:
        #
        #    img = Image.fromarray(obj)
        #    img = img.convert(mode)
        #
        # The resulting image has very different actual pixel values than before.
        #
        # TODO: Issue #9151. Pillow has an open PR to restore the functionality
        #       we rely on:
        #
        #       https://github.com/python-pillow/Pillow/pull/9063
        #
        #       When that is part of a release, we can revisit this hack below.
        arr = obj.__array_interface__
        shape = arr["shape"]
        ndim = len(shape)
        size = 1 if ndim == 1 else shape[1], shape[0]

        strides = arr.get("strides", None)
        contiguous_obj: Union[np.ndarray, bytes] = obj
        if strides is not None:
            # We require that the data is contiguous; if it is not, we need to
            # convert it into a contiguous format.
            if hasattr(obj, "tobytes"):
                contiguous_obj = obj.tobytes()
            elif hasattr(obj, "tostring"):
                contiguous_obj = obj.tostring()
            else:
                raise ValueError("Unable to convert obj into contiguous format")

        return Image.frombuffer(mode, size, contiguous_obj, "raw", mode, 0, 1)
    else:
        return Image.fromarray(obj, mode)


@torch.no_grad()
def save_image(
    tensor: Union[torch.Tensor, list[torch.Tensor]],
    fp: Union[str, pathlib.Path, BinaryIO],
    format: Optional[str] = None,
    **kwargs,
) -> None:
    """
    Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        fp (string or file object): A filename or a file object
        format(Optional):  If omitted, the format to use is determined from the filename extension.
            If a file object was used instead of a filename, this parameter should always be used.
        **kwargs: Other arguments are documented in ``make_grid``.
    """

    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(save_image)
    grid = make_grid(tensor, **kwargs)
    # Add 0.5 after unnormalizing to [0, 255] to round to the nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(fp, format=format)


@torch.no_grad()
def draw_bounding_boxes(
    image: torch.Tensor,
    boxes: torch.Tensor,
    labels: Optional[list[str]] = None,
    colors: Optional[Union[list[Union[str, tuple[int, int, int]]], str, tuple[int, int, int]]] = None,
    fill: Optional[bool] = False,
    width: int = 1,
    font: Optional[str] = None,
    font_size: Optional[int] = None,
    label_colors: Optional[Union[list[Union[str, tuple[int, int, int]]], str, tuple[int, int, int]]] = None,
    fill_labels: bool = False,
) -> torch.Tensor:
    """
    Draws bounding boxes on given RGB image.
    The image values should be uint8 in [0, 255] or float in [0, 1].
    If fill is True, Resulting Tensor should be saved as PNG image.

    Args:
        image (Tensor): Tensor of shape (C, H, W) and dtype uint8 or float.
        boxes (Tensor): Tensor of size (N, 4) or (N, 8) containing bounding boxes.
            For (N, 4), the format is (xmin, ymin, xmax, ymax) and the boxes are absolute coordinates with respect to the image.
            In other words: `0 <= xmin < xmax < W` and `0 <= ymin < ymax < H`.
            For (N, 8), the format is (x1, y1, x2, y2, x3, y3, x4, y4) and the boxes are absolute coordinates with respect to the underlying
            object, so no need to verify the latter inequalities.
        labels (List[str]): List containing the labels of bounding boxes.
        colors (color or list of colors, optional): List containing the colors
            of the boxes or single color for all boxes. The color can be represented as
            PIL strings e.g. "red" or "#FF00FF", or as RGB tuples e.g. ``(240, 10, 157)``.
            By default, random colors are generated for boxes.
        fill (bool): If `True` fills the bounding box with specified color.
        width (int): Width of bounding box.
        font (str): A filename containing a TrueType font. If the file is not found in this filename, the loader may
            also search in other directories, such as the `fonts/` directory on Windows or `/Library/Fonts/`,
            `/System/Library/Fonts/` and `~/Library/Fonts/` on macOS.
        font_size (int): The requested font size in points.
        label_colors (color or list of colors, optional): Colors for the label text.  See the description of the
            `colors` argument for details.  Defaults to the same colors used for the boxes, or to black if ``fill_labels`` is True.
        fill_labels (bool): If `True` fills the label background with specified box color (from the ``colors`` parameter). Default: False.

    Returns:
        img (Tensor[C, H, W]): Image Tensor of dtype uint8 with bounding boxes plotted.

    """
    import torchvision.transforms.v2.functional as F  # noqa

    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(draw_bounding_boxes)
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Tensor expected, got {type(image)}")
    elif not (image.dtype == torch.uint8 or image.is_floating_point()):
        raise ValueError(f"The image dtype must be uint8 or float, got {image.dtype}")
    elif image.dim() != 3:
        raise ValueError("Pass individual images, not batches")
    elif image.size(0) not in {1, 3}:
        raise ValueError("Only grayscale and RGB images are supported")
    elif boxes.shape[-1] == 4 and ((boxes[:, 0] > boxes[:, 2]).any() or (boxes[:, 1] > boxes[:, 3]).any()):
        raise ValueError(
            "Boxes need to be in (xmin, ymin, xmax, ymax) format. Use torchvision.ops.box_convert to convert them"
        )

    num_boxes = boxes.shape[0]

    if num_boxes == 0:
        warnings.warn("boxes doesn't contain any box. No box was drawn")
        return image

    if labels is None:
        labels: Union[list[str], list[None]] = [None] * num_boxes  # type: ignore[no-redef]
    elif len(labels) != num_boxes:
        raise ValueError(
            f"Number of boxes ({num_boxes}) and labels ({len(labels)}) mismatch. Please specify labels for each box."
        )

    colors = _parse_colors(colors, num_objects=num_boxes)
    if label_colors or fill_labels:
        label_colors = _parse_colors(label_colors if label_colors else "black", num_objects=num_boxes)  # type: ignore[assignment]
    else:
        label_colors = colors.copy()  # type: ignore[assignment]

    if font is None:
        if font_size is not None:
            warnings.warn("Argument 'font_size' will be ignored since 'font' is not set.")
        txt_font = ImageFont.load_default()
    else:
        txt_font = ImageFont.truetype(font=font, size=font_size or 10)

    # Handle Grayscale images
    if image.size(0) == 1:
        image = torch.tile(image, (3, 1, 1))

    original_dtype = image.dtype
    if original_dtype.is_floating_point:
        image = F.to_dtype(image, dtype=torch.uint8, scale=True)

    img_to_draw = F.to_pil_image(image)
    img_boxes = boxes.to(torch.int64).tolist()

    if fill:
        draw = _ImageDrawTV(img_to_draw, "RGBA")
    else:
        draw = _ImageDrawTV(img_to_draw)

    for bbox, color, label, label_color in zip(img_boxes, colors, labels, label_colors):  # type: ignore[arg-type]
        draw_method = draw.oriented_rectangle if len(bbox) > 4 else draw.rectangle
        fill_color = color + (100,) if fill else None
        draw_method(bbox, width=width, outline=color, fill=fill_color)

        if label is not None:
            box_margin = 1
            margin = width + box_margin
            if fill_labels:
                left, top, right, bottom = draw.textbbox((bbox[0] + margin, bbox[1] + margin), label, font=txt_font)
                draw.rectangle(
                    (left - box_margin, top - box_margin, right + box_margin, bottom + box_margin), fill=color
                )
            draw.text((bbox[0] + margin, bbox[1] + margin), label, fill=label_color, font=txt_font)  # type: ignore[arg-type]

    out = F.pil_to_tensor(img_to_draw)
    if original_dtype.is_floating_point:
        out = F.to_dtype(out, dtype=original_dtype, scale=True)
    return out


@torch.no_grad()
def draw_segmentation_masks(
    image: torch.Tensor,
    masks: torch.Tensor,
    alpha: float = 0.8,
    colors: Optional[Union[list[Union[str, tuple[int, int, int]]], str, tuple[int, int, int]]] = None,
) -> torch.Tensor:
    """
    Draws segmentation masks on given RGB image.
    The image values should be uint8 in [0, 255] or float in [0, 1].

    Args:
        image (Tensor): Tensor of shape (3, H, W) and dtype uint8 or float.
        masks (Tensor): Tensor of shape (num_masks, H, W) or (H, W) and dtype bool.
        alpha (float): Float number between 0 and 1 denoting the transparency of the masks.
            0 means full transparency, 1 means no transparency.
        colors (color or list of colors, optional): List containing the colors
            of the masks or single color for all masks. The color can be represented as
            PIL strings e.g. "red" or "#FF00FF", or as RGB tuples e.g. ``(240, 10, 157)``.
            By default, random colors are generated for each mask.

    Returns:
        img (Tensor[C, H, W]): Image Tensor, with segmentation masks drawn on top.
    """

    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(draw_segmentation_masks)
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"The image must be a tensor, got {type(image)}")
    elif not (image.dtype == torch.uint8 or image.is_floating_point()):
        raise ValueError(f"The image dtype must be uint8 or float, got {image.dtype}")
    elif image.dim() != 3:
        raise ValueError("Pass individual images, not batches")
    elif image.size()[0] != 3:
        raise ValueError("Pass an RGB image. Other Image formats are not supported")
    if masks.ndim == 2:
        masks = masks[None, :, :]
    if masks.ndim != 3:
        raise ValueError("masks must be of shape (H, W) or (batch_size, H, W)")
    if masks.dtype != torch.bool:
        raise ValueError(f"The masks must be of dtype bool. Got {masks.dtype}")
    if masks.shape[-2:] != image.shape[-2:]:
        raise ValueError("The image and the masks must have the same height and width")

    num_masks = masks.size()[0]
    overlapping_masks = masks.sum(dim=0) > 1

    if num_masks == 0:
        warnings.warn("masks doesn't contain any mask. No mask was drawn")
        return image

    original_dtype = image.dtype
    colors = [
        torch.tensor(color, dtype=original_dtype, device=image.device)
        for color in _parse_colors(colors, num_objects=num_masks, dtype=original_dtype)
    ]

    img_to_draw = image.detach().clone()
    # TODO: There might be a way to vectorize this
    for mask, color in zip(masks, colors):
        img_to_draw[:, mask] = color[:, None]

    img_to_draw[:, overlapping_masks] = 0

    out = image * (1 - alpha) + img_to_draw * alpha
    # Note: at this point, out is a float tensor in [0, 1] or [0, 255] depending on original_dtype
    return out.to(original_dtype)


@torch.no_grad()
def draw_keypoints(
    image: torch.Tensor,
    keypoints: torch.Tensor,
    connectivity: Optional[list[tuple[int, int]]] = None,
    colors: Optional[Union[str, tuple[int, int, int]]] = None,
    radius: int = 2,
    width: int = 3,
    visibility: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Draws Keypoints on given RGB image.
    The image values should be uint8 in [0, 255] or float in [0, 1].
    Keypoints can be drawn for multiple instances at a time.

    This method allows that keypoints and their connectivity are drawn based on the visibility of this keypoint.

    Args:
        image (Tensor): Tensor of shape (3, H, W) and dtype uint8 or float.
        keypoints (Tensor): Tensor of shape (num_instances, K, 2) the K keypoint locations for each of the N instances,
            in the format [x, y].
        connectivity (List[Tuple[int, int]]]): A List of tuple where each tuple contains a pair of keypoints
            to be connected.
            If at least one of the two connected keypoints has a ``visibility`` of False,
            this specific connection is not drawn.
            Exclusions due to invisibility are computed per-instance.
        colors (str, Tuple): The color can be represented as
            PIL strings e.g. "red" or "#FF00FF", or as RGB tuples e.g. ``(240, 10, 157)``.
        radius (int): Integer denoting radius of keypoint.
        width (int): Integer denoting width of line connecting keypoints.
        visibility (Tensor): Tensor of shape (num_instances, K) specifying the visibility of the K
            keypoints for each of the N instances.
            True means that the respective keypoint is visible and should be drawn.
            False means invisible, so neither the point nor possible connections containing it are drawn.
            The input tensor will be cast to bool.
            Default ``None`` means that all the keypoints are visible.
            For more details, see :ref:`draw_keypoints_with_visibility`.

    Returns:
        img (Tensor[C, H, W]): Image Tensor with keypoints drawn.
    """

    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(draw_keypoints)
    # validate image
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"The image must be a tensor, got {type(image)}")
    elif not (image.dtype == torch.uint8 or image.is_floating_point()):
        raise ValueError(f"The image dtype must be uint8 or float, got {image.dtype}")
    elif image.dim() != 3:
        raise ValueError("Pass individual images, not batches")
    elif image.size()[0] != 3:
        raise ValueError("Pass an RGB image. Other Image formats are not supported")

    # validate keypoints
    if keypoints.ndim != 3:
        raise ValueError("keypoints must be of shape (num_instances, K, 2)")

    # validate visibility
    if visibility is None:  # set default
        visibility = torch.ones(keypoints.shape[:-1], dtype=torch.bool)
    if visibility.ndim == 3:
        # If visibility was passed as pred.split([2, 1], dim=-1), it will be of shape (num_instances, K, 1).
        # We make sure it is of shape (num_instances, K). This isn't documented, we're just being nice.
        visibility = visibility.squeeze(-1)
    if visibility.ndim != 2:
        raise ValueError(f"visibility must be of shape (num_instances, K). Got ndim={visibility.ndim}")
    if visibility.shape != keypoints.shape[:-1]:
        raise ValueError(
            "keypoints and visibility must have the same dimensionality for num_instances and K. "
            f"Got {visibility.shape = } and {keypoints.shape = }"
        )

    original_dtype = image.dtype
    if original_dtype.is_floating_point:
        from torchvision.transforms.v2.functional import to_dtype  # noqa

        image = to_dtype(image, dtype=torch.uint8, scale=True)

    ndarr = image.permute(1, 2, 0).cpu().numpy()
    img_to_draw = Image.fromarray(ndarr)
    draw = ImageDraw.Draw(img_to_draw)
    img_kpts = keypoints.to(torch.int64).tolist()
    img_vis = visibility.cpu().bool().tolist()

    for kpt_inst, vis_inst in zip(img_kpts, img_vis):
        for kpt_coord, kp_vis in zip(kpt_inst, vis_inst):
            if not kp_vis:
                continue
            x1 = kpt_coord[0] - radius
            x2 = kpt_coord[0] + radius
            y1 = kpt_coord[1] - radius
            y2 = kpt_coord[1] + radius
            draw.ellipse([x1, y1, x2, y2], fill=colors, outline=None, width=0)

        if connectivity:
            for connection in connectivity:
                if (not vis_inst[connection[0]]) or (not vis_inst[connection[1]]):
                    continue
                start_pt_x = kpt_inst[connection[0]][0]
                start_pt_y = kpt_inst[connection[0]][1]

                end_pt_x = kpt_inst[connection[1]][0]
                end_pt_y = kpt_inst[connection[1]][1]

                draw.line(
                    ((start_pt_x, start_pt_y), (end_pt_x, end_pt_y)),
                    width=width,
                )

    out = torch.from_numpy(np.array(img_to_draw)).permute(2, 0, 1)
    if original_dtype.is_floating_point:
        out = to_dtype(out, dtype=original_dtype, scale=True)
    return out


# Flow visualization code adapted from https://github.com/tomrunia/OpticalFlow_Visualization
@torch.no_grad()
def flow_to_image(flow: torch.Tensor) -> torch.Tensor:
    """
    Converts a flow to an RGB image.

    Args:
        flow (Tensor): Flow of shape (N, 2, H, W) or (2, H, W) and dtype torch.float.

    Returns:
        img (Tensor): Image Tensor of dtype uint8 where each color corresponds
            to a given flow direction. Shape is (N, 3, H, W) or (3, H, W) depending on the input.
    """

    if flow.dtype != torch.float:
        raise ValueError(f"Flow should be of dtype torch.float, got {flow.dtype}.")

    orig_shape = flow.shape
    if flow.ndim == 3:
        flow = flow[None]  # Add batch dim

    if flow.ndim != 4 or flow.shape[1] != 2:
        raise ValueError(f"Input flow should have shape (2, H, W) or (N, 2, H, W), got {orig_shape}.")

    max_norm = torch.sum(flow**2, dim=1).sqrt().max()
    epsilon = torch.finfo((flow).dtype).eps
    normalized_flow = flow / (max_norm + epsilon)
    img = _normalized_flow_to_image(normalized_flow)

    if len(orig_shape) == 3:
        img = img[0]  # Remove batch dim
    return img


@torch.no_grad()
def _normalized_flow_to_image(normalized_flow: torch.Tensor) -> torch.Tensor:
    """
    Converts a batch of normalized flow to an RGB image.

    Args:
        normalized_flow (torch.Tensor): Normalized flow tensor of shape (N, 2, H, W)
    Returns:
       img (Tensor(N, 3, H, W)): Flow visualization image of dtype uint8.
    """

    N, _, H, W = normalized_flow.shape
    device = normalized_flow.device
    flow_image = torch.zeros((N, 3, H, W), dtype=torch.uint8, device=device)
    colorwheel = _make_colorwheel().to(device)  # shape [55x3]
    num_cols = colorwheel.shape[0]
    norm = torch.sum(normalized_flow**2, dim=1).sqrt()
    a = torch.atan2(-normalized_flow[:, 1, :, :], -normalized_flow[:, 0, :, :]) / torch.pi
    fk = (a + 1) / 2 * (num_cols - 1)
    k0 = torch.floor(fk).to(torch.long)
    k1 = k0 + 1
    k1[k1 == num_cols] = 0
    f = fk - k0

    for c in range(colorwheel.shape[1]):
        tmp = colorwheel[:, c]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1
        col = 1 - norm * (1 - col)
        flow_image[:, c, :, :] = torch.floor(255 * col)
    return flow_image


def _make_colorwheel() -> torch.Tensor:
    """
    Generates a color wheel for optical flow visualization as presented in:
    Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
    URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf.

    Returns:
        colorwheel (Tensor[55, 3]): Colorwheel Tensor.
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = torch.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = torch.floor(255 * torch.arange(0, RY) / RY)
    col = col + RY
    # YG
    colorwheel[col : col + YG, 0] = 255 - torch.floor(255 * torch.arange(0, YG) / YG)
    colorwheel[col : col + YG, 1] = 255
    col = col + YG
    # GC
    colorwheel[col : col + GC, 1] = 255
    colorwheel[col : col + GC, 2] = torch.floor(255 * torch.arange(0, GC) / GC)
    col = col + GC
    # CB
    colorwheel[col : col + CB, 1] = 255 - torch.floor(255 * torch.arange(CB) / CB)
    colorwheel[col : col + CB, 2] = 255
    col = col + CB
    # BM
    colorwheel[col : col + BM, 2] = 255
    colorwheel[col : col + BM, 0] = torch.floor(255 * torch.arange(0, BM) / BM)
    col = col + BM
    # MR
    colorwheel[col : col + MR, 2] = 255 - torch.floor(255 * torch.arange(MR) / MR)
    colorwheel[col : col + MR, 0] = 255
    return colorwheel


def _generate_color_palette(num_objects: int):
    palette = torch.tensor([2**25 - 1, 2**15 - 1, 2**21 - 1])
    return [tuple((i * palette) % 255) for i in range(num_objects)]


def _parse_colors(
    colors: Union[None, str, tuple[int, int, int], list[Union[str, tuple[int, int, int]]]],
    *,
    num_objects: int,
    dtype: torch.dtype = torch.uint8,
) -> list[tuple[int, int, int]]:
    """
    Parses a specification of colors for a set of objects.

    Args:
        colors: A specification of colors for the objects. This can be one of the following:
            - None: to generate a color palette automatically.
            - A list of colors: where each color is either a string (specifying a named color) or an RGB tuple.
            - A string or an RGB tuple: to use the same color for all objects.

            If `colors` is a tuple, it should be a 3-tuple specifying the RGB values of the color.
            If `colors` is a list, it should have at least as many elements as the number of objects to color.

        num_objects (int): The number of objects to color.

    Returns:
        A list of 3-tuples, specifying the RGB values of the colors.

    Raises:
        ValueError: If the number of colors in the list is less than the number of objects to color.
                    If `colors` is not a list, tuple, string or None.
    """
    if colors is None:
        colors = _generate_color_palette(num_objects)
    elif isinstance(colors, list):
        if len(colors) < num_objects:
            raise ValueError(
                f"Number of colors must be equal or larger than the number of objects, but got {len(colors)} < {num_objects}."
            )
    elif not isinstance(colors, (tuple, str)):
        raise ValueError(f"`colors` must be a tuple or a string, or a list thereof, but got {colors}.")
    elif isinstance(colors, tuple) and len(colors) != 3:
        raise ValueError(f"If passed as tuple, colors should be an RGB triplet, but got {colors}.")
    else:  # colors specifies a single color for all objects
        colors = [colors] * num_objects

    colors = [ImageColor.getrgb(color) if isinstance(color, str) else color for color in colors]
    if dtype.is_floating_point:  # [0, 255] -> [0, 1]
        colors = [tuple(v / 255 for v in color) for color in colors]  # type: ignore[union-attr]
    return colors  # type: ignore[return-value]


def _log_api_usage_once(obj: Any) -> None:
    """
    Logs API usage(module and name) within an organization.
    In a large ecosystem, it's often useful to track the PyTorch and
    TorchVision APIs usage. This API provides the similar functionality to the
    logging module in the Python stdlib. It can be used for debugging purpose
    to log which methods are used and by default it is inactive, unless the user
    manually subscribes a logger via the `SetAPIUsageLogger method <https://github.com/pytorch/pytorch/blob/eb3b9fe719b21fae13c7a7cf3253f970290a573e/c10/util/Logging.cpp#L114>`_.
    Please note it is triggered only once for the same API call within a process.
    It does not collect any data from open-source users since it is no-op by default.
    For more information, please refer to
    * PyTorch note: https://pytorch.org/docs/stable/notes/large_scale_deployments.html#api-usage-logging;
    * Logging policy: https://github.com/pytorch/vision/issues/5052;

    Args:
        obj (class instance or method): an object to extract info from.
    """
    module = obj.__module__
    if not module.startswith("torchvision"):
        module = f"torchvision.internal.{module}"
    name = obj.__class__.__name__
    if isinstance(obj, FunctionType):
        name = obj.__name__
    torch._C._log_api_usage_once(f"{module}.{name}")


def _make_ntuple(x: Any, n: int) -> tuple[Any, ...]:
    """
    Make n-tuple from input x. If x is an iterable, then we just convert it to tuple.
    Otherwise, we will make a tuple of length n, all with value of x.
    reference: https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/utils.py#L8

    Args:
        x (Any): input value
        n (int): length of the resulting tuple
    """
    if isinstance(x, collections.abc.Iterable):
        return tuple(x)
    return tuple(repeat(x, n))
