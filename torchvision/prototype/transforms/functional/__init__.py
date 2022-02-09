from torchvision.transforms import InterpolationMode

from ._augment import (
    erase_image,
    erase,
    mixup_image,
    mixup_one_hot_label,
    mixup,
    cutmix_image,
    cutmix_one_hot_label,
    cutmix,
)
from ._color import (
    adjust_brightness_image,
    adjust_brightness,
    adjust_contrast_image,
    adjust_contrast,
    adjust_saturation_image,
    adjust_saturation,
    adjust_sharpness_image,
    adjust_sharpness,
    posterize_image,
    posterize,
    solarize_image,
    solarize,
    autocontrast_image,
    autocontrast,
    equalize_image,
    equalize,
    invert_image,
    invert,
)
from ._geometry import (
    horizontal_flip_bounding_box,
    horizontal_flip_image,
    horizontal_flip,
    resize_bounding_box,
    resize_image,
    resize_segmentation_mask,
    resize,
    center_crop_image,
    center_crop,
    resized_crop_image,
    resized_crop,
    affine_image,
    affine,
    rotate_image,
    rotate,
)
from ._meta_conversion import convert_color_space, convert_bounding_box_format
from ._misc import normalize_image, normalize
from ._type_conversion import decode_image_with_pil, decode_video_with_av, label_to_one_hot
