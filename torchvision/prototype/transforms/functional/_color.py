from torchvision.transforms import functional_tensor as _FT

from .utils import _from_legacy_kernel


adjust_brightness_image = _from_legacy_kernel(_FT.adjust_brightness)

adjust_saturation_image = _from_legacy_kernel(_FT.adjust_saturation)

adjust_contrast_image = _from_legacy_kernel(_FT.adjust_contrast)

adjust_sharpness_image = _from_legacy_kernel(_FT.adjust_sharpness)

posterize_image = _from_legacy_kernel(_FT.posterize)

solarize_image = _from_legacy_kernel(_FT.solarize)

autocontrast_image = _from_legacy_kernel(_FT.autocontrast)

equalize_image = _from_legacy_kernel(_FT.equalize)

invert_image = _from_legacy_kernel(_FT.invert)
