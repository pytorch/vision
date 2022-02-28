# He should create an issue that lists the steps that need to be performed for rolling out the API to main TorchVision.
# I got a similar for the models, see here: https://github.com/pytorch/vision/issues/4679
# One of the key things we would need to take care of is that all the kernels below will need logging. This is because
# there will be no high-level kernel (like `F` on main) and we would instead need to put tracking directly in each
# low-level kernels which will be now public (now functional_pil|tensor are private).

from torchvision.transforms import InterpolationMode  # usort: skip
from ._meta import (
    convert_bounding_box_format,
    convert_image_color_space_tensor,
    convert_image_color_space_pil,
)  # usort: skip

from ._augment import (
    erase_image_tensor,
    mixup_image_tensor,
    mixup_one_hot_label,
    cutmix_image_tensor,
    cutmix_one_hot_label,
)
from ._color import (
    adjust_brightness_image_tensor,
    adjust_brightness_image_pil,
    adjust_contrast_image_tensor,
    adjust_contrast_image_pil,
    adjust_saturation_image_tensor,
    adjust_saturation_image_pil,
    adjust_sharpness_image_tensor,
    adjust_sharpness_image_pil,
    posterize_image_tensor,
    posterize_image_pil,
    solarize_image_tensor,
    solarize_image_pil,
    autocontrast_image_tensor,
    autocontrast_image_pil,
    equalize_image_tensor,
    equalize_image_pil,
    invert_image_tensor,
    invert_image_pil,
    adjust_hue_image_tensor,
    adjust_hue_image_pil,
    adjust_gamma_image_tensor,
    adjust_gamma_image_pil,
)
from ._geometry import (
    horizontal_flip_bounding_box,
    horizontal_flip_image_tensor,
    horizontal_flip_image_pil,
    resize_bounding_box,
    resize_image_tensor,
    resize_image_pil,
    resize_segmentation_mask,
    center_crop_image_tensor,
    center_crop_image_pil,
    resized_crop_image_tensor,
    resized_crop_image_pil,
    affine_image_tensor,
    affine_image_pil,
    rotate_image_tensor,
    rotate_image_pil,
    pad_image_tensor,
    pad_image_pil,
    crop_image_tensor,
    crop_image_pil,
    perspective_image_tensor,
    perspective_image_pil,
    vertical_flip_image_tensor,
    vertical_flip_image_pil,
)
from ._misc import normalize_image_tensor, gaussian_blur_image_tensor
from ._type_conversion import decode_image_with_pil, decode_video_with_av, label_to_one_hot

# What are the migration plans for public methods without new API equivalents? There are two categories:
# 1. Deprecated methods which have equivalents on the new API (_legacy.py?):
# - get_image_size, get_image_num_channels: use get_dimensions_image_tensor|pil
# - to_grayscale, rgb_to_grayscale: use convert_image_color_space_tensor|pil
# 2. Those without equivalents on the new API:
# - five_crop, ten_crop (must be added)
# - pil_to_tensor, to_pil_image (_legacy.py?)
# - to_tensor() (deprecate vfdev-5?)
# We need a plan for both categories implemented on the new API.
